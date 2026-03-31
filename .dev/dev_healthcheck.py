"""
.dev/dev_healthcheck.py — architectural linter.

Performs static checks on the codebase to enforce key boundaries:

  1. DB driver imports — only databank/ may import sqlite3 / aiosqlite / psycopg2.
  2. ComfyUI imports — only extractor/ and comfyui/ (and root __init__.py) may import them.
  3. Hard-coded absolute paths — only core/paths.py may contain them.
  4. print() / logging calls — no module outside .dev/ or lab/ may call print() directly
     or import Python's standard 'logging' module.  Use emit() from core.diagnostics.
  5. Star imports — 'from X import *' is forbidden everywhere (explicit > implicit).
  6. Module import allowlist — cross-module imports are forbidden unless explicitly allowed.
  7. Private cross-module imports — no module imports _private symbols from another module.
  8. ARCH.CHEAT cleanup context — every ARCH.CHEAT emit() must include cleanup=... kwarg.
  9. Diagnostic code registry — all emit() codes must be registered in contracts/diagnostic_codes.py.
 10. Contracts / review metadata integrity — cross-validates contracts/ surfaces and review payload.
 11. Aliased emit imports — 'from core.diagnostics import emit as X' is forbidden; import as
     'emit' directly so check 9 can see all emit() call sites.

--- HOW TO LOG / REPORT ERRORS IN THIS PROJECT ---

  WRONG:  print("something happened")
  WRONG:  import logging; logging.error("something")
  WRONG:  raise RuntimeError("something")  (as the only error record)
  WRONG:  from core.diagnostics import emit as _emit  (alias hides calls from check 9)

  RIGHT:  from core.diagnostics import emit
          emit("WARN", "MY.CODE", "module.function", "human message", key=value)

  emit() writes a JSON record to logs_root/diagnostics.jsonl and echoes WARN+ to stderr.
  Every code string must be registered in contracts/diagnostic_codes.py first.
  See core/diagnostics.py for the full emit() signature.

Exit 0 = clean (zero violations).
Exit 1 = violations found.

Usage:
    python .dev/dev_healthcheck.py
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root and exclusion rules
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Directories to skip entirely during scanning
_SKIP_DIRS = {".git", ".docs", ".agents", "__pycache__", ".venv", "venv", "node_modules"}

# DB driver module names that only databank/ is allowed to import
_DB_DRIVERS = {"sqlite3", "aiosqlite", "psycopg2", "psycopg", "pymysql", "cx_Oracle"}

# ComfyUI module name prefixes that only extractor/ and comfyui/ may import
_COMFYUI_PREFIXES = ("comfy", "nodes", "folder_paths", "execution", "server")

# Allowed to contain ComfyUI imports (relative to project root)
_COMFYUI_ALLOWED = {"extractor", "comfyui", "lab"}

# Only this file is allowed to contain hard-coded absolute paths
_PATHS_MODULE_REL = Path("core") / "paths.py"

# Modules allowed to use print() directly (dev tools, the diagnostics module, lab/, and
# databank/operations/ migration scripts).
# lab/ is off-grid exploration tooling — same category as .dev/ — not production pipeline code.
# databank/operations/ scripts are operator-facing one-off maintenance tools; they need
# terminal output for the same reason .dev/ scripts do.
_PRINT_ALLOWED = {".dev", "lab", "databank/operations", "core/diagnostics.py"}

# All first-party top-level module names (used for cross-module checks).
# MAINTENANCE RULE: when a new top-level module is created, add it here — no human
# permission needed. If a module is missing from this list the boundary checker is
# completely blind to its imports.
_OWN_MODULES: frozenset[str] = frozenset({
    "extractor", "bouncer", "databank", "reader",
    "command_center", "core", "contracts", "lab", "comfyui", "operator_app",
})

# ---------------------------------------------------------------------------
# Allowed module import pairs (Boundary Door Law)
#
# There is no default permit. For cross-module imports between known first-party
# modules, silence means forbidden.
#
# OPEN DOORS (importable by anyone, not listed here — handled in _is_allowed_cross_module_import):
#   core/, contracts/, databank.treasurer
#
# PERMISSION RULE: adding a new pair to this list requires explicit human permission.
# Do not add pairs on your own judgment, even temporarily. Use ARCH.CHEAT if you
# need to acknowledge a temporary violation while waiting for approval.
#
# .dev/ and the root __init__.py are exempt (test scripts + ComfyUI loader glue).
# ---------------------------------------------------------------------------

_ALLOWED_IMPORT_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("operator_app", "command_center"),
    ("comfyui", "extractor"),
    ("extractor", "bouncer"),
    ("comfyui", "lab"),
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_py_files() -> list[Path]:
    files = []
    for path in _PROJECT_ROOT.rglob("*.py"):
        rel = path.relative_to(_PROJECT_ROOT)
        parts = rel.parts
        if any(part in _SKIP_DIRS for part in parts):
            continue
        files.append(path)
    return sorted(files)


def _rel(path: Path) -> str:
    return str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")


def _parse(path: Path) -> ast.Module | None:
    try:
        source = path.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None  # syntax errors are reported separately if needed


def _imports_in(tree: ast.Module) -> list[tuple[str, int]]:
    """Return (module_name, lineno) for all import statements in the AST."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            results.append((mod, node.lineno))
    return results


def _emit_calls_in(tree: ast.Module) -> list[tuple[str, int]]:
    """Return (code_arg, lineno) for all emit() calls found in the AST.

    Only extracts the 'code' argument when it is a plain string literal
    (i.e. emit(severity, "SOME.CODE", ...)).  Variable references are skipped.
    """
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_emit = (isinstance(func, ast.Name) and func.id == "emit") or (
            isinstance(func, ast.Attribute) and func.attr == "emit"
        )
        if not is_emit:
            continue
        # code is the 2nd positional argument (index 1)
        if len(node.args) >= 2:
            code_node = node.args[1]
            if isinstance(code_node, ast.Constant) and isinstance(code_node.value, str):
                results.append((code_node.value, node.lineno))
        # or keyword argument code=...
        for kw in node.keywords:
            if kw.arg == "code" and isinstance(kw.value, ast.Constant):
                results.append((kw.value.value, node.lineno))
    return results


def _print_calls_in(tree: ast.Module) -> list[int]:
    """Return line numbers of all print() calls in the AST."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                lines.append(node.lineno)
    return lines


def _has_absolute_string_path(tree: ast.Module) -> list[int]:
    """Return line numbers of string literals that look like absolute file paths."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            # Unix absolute path
            if val.startswith("/") and len(val) > 1 and val[1].isalpha():
                lines.append(node.lineno)
            # Windows absolute path: C:/ or C:\
            elif len(val) >= 3 and val[1] == ":" and val[2] in ("/", "\\"):
                lines.append(node.lineno)
    return lines


def _load_diagnostic_codes() -> set[str] | None:
    """Load the central diagnostic code registry if it exists."""
    codes_file = _PROJECT_ROOT / "contracts" / "diagnostic_codes.py"
    if not codes_file.exists():
        return None
    try:
        source = codes_file.read_text(encoding="utf-8")
        tree = ast.parse(source)

        def _extract_set(val_node) -> set[str] | None:
            if isinstance(val_node, (ast.Set, ast.List, ast.Tuple)):
                codes: set[str] = set()
                for elt in val_node.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        codes.add(elt.value)
                return codes
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("ALL_CODES", "CODES"):
                        result = _extract_set(node.value)
                        if result is not None:
                            return result
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id in ("ALL_CODES", "CODES"):
                    if node.value is not None:
                        result = _extract_set(node.value)
                        if result is not None:
                            return result
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_db_drivers(files: list[Path]) -> list[str]:
    """Violation if a DB driver is imported outside databank/.

    .dev/ scripts are exempt: they are maintenance/migration tooling that may
    need direct DB access as an ARCH.CHEAT (documented by emit() annotations).
    The ARCH.CHEAT check enforces that such scripts carry a cleanup= kwarg.
    """
    violations = []
    for path in files:
        parts = path.relative_to(_PROJECT_ROOT).parts
        if parts and parts[0] in ("databank", ".dev"):
            continue  # allowed
        tree = _parse(path)
        if tree is None:
            continue
        for mod, lineno in _imports_in(tree):
            top = mod.split(".")[0]
            if top in _DB_DRIVERS:
                violations.append(
                    f"  {_rel(path)}:{lineno} — imports DB driver '{mod}' outside databank/"
                )
    return violations


def check_comfyui_imports(files: list[Path]) -> list[str]:
    """Violation if a ComfyUI module is imported outside extractor/, comfyui/, lab/."""
    violations = []
    for path in files:
        parts = path.relative_to(_PROJECT_ROOT).parts
        if parts and parts[0] in _COMFYUI_ALLOWED:
            continue  # allowed directory
        # Root __init__.py is the ComfyUI package entry point — always allowed.
        if parts == ("__init__.py",):
            continue
        tree = _parse(path)
        if tree is None:
            continue
        for mod, lineno in _imports_in(tree):
            top = mod.split(".")[0]
            if any(top.startswith(prefix) for prefix in _COMFYUI_PREFIXES):
                violations.append(
                    f"  {_rel(path)}:{lineno} — imports ComfyUI module '{mod}' outside allowed folders"
                )
    return violations


def check_hardcoded_paths(files: list[Path]) -> list[str]:
    """Violation if an absolute path string literal appears outside core/paths.py."""
    violations = []
    allowed = _PROJECT_ROOT / _PATHS_MODULE_REL
    for path in files:
        if path.resolve() == allowed.resolve():
            continue
        tree = _parse(path)
        if tree is None:
            continue
        for lineno in _has_absolute_string_path(tree):
            violations.append(
                f"  {_rel(path)}:{lineno} — hard-coded absolute path string outside core/paths.py"
            )
    return violations


def check_print_statements(files: list[Path]) -> list[str]:
    """Violation if print() is called outside .dev/, lab/, databank/operations/, or core/diagnostics.py.

    Fix: use emit() from core.diagnostics instead.
    emit("WARN", "MY.CODE", "module.function", "human message", key=value)
    Codes must be registered in contracts/diagnostic_codes.py.
    """
    violations = []
    for path in files:
        rel = _rel(path)
        parts = path.relative_to(_PROJECT_ROOT).parts
        if parts and parts[0] in (".dev", "lab"):
            continue
        if len(parts) >= 2 and parts[0] == "databank" and parts[1] == "operations":
            continue  # operator migration scripts — same category as .dev/
        if rel == "core/diagnostics.py":
            continue
        tree = _parse(path)
        if tree is None:
            continue
        for lineno in _print_calls_in(tree):
            violations.append(
                f"  {rel}:{lineno} — print() call; use emit() from core.diagnostics instead"
                f" (only .dev/, lab/, databank/operations/, core/diagnostics.py may use print)"
            )
    return violations


def check_logging_module_usage(files: list[Path]) -> list[str]:
    """Violation if Python's standard 'logging' module is imported anywhere in production code.

    This project uses emit() from core.diagnostics for all structured logging.
    Python's logging module is not the project mechanism and bypasses the diagnostic
    record trail (diagnostics.jsonl) entirely.

    Fix: from core.diagnostics import emit
         emit("WARN", "MY.CODE", "module.function", "message", key=value)
    Codes must be registered in contracts/diagnostic_codes.py.
    """
    violations = []
    for path in files:
        rel = _rel(path)
        parts = path.relative_to(_PROJECT_ROOT).parts
        # .dev/ uses print() directly (already exempt from print check); logging also exempt there
        if parts and parts[0] == ".dev":
            continue
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "logging" or alias.name.startswith("logging."):
                        violations.append(
                            f"  {rel}:{node.lineno} — 'import logging';"
                            f" use emit() from core.diagnostics instead (see healthcheck docstring)"
                        )
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").startswith("logging"):
                    violations.append(
                        f"  {rel}:{node.lineno} — import from 'logging' module;"
                        f" use emit() from core.diagnostics instead (see healthcheck docstring)"
                    )
    return violations


def check_aliased_emit_imports(files: list[Path]) -> list[str]:
    """Violation if core.diagnostics.emit is imported under any alias other than 'emit'.

    Aliases like '_emit' are invisible to the AST scanner in _emit_calls_in(), which
    only finds calls whose function name is literally 'emit'.  An aliased import
    silently hides every emit() call in that file from check 9 (diagnostic code
    registry), allowing unregistered codes to slip through undetected.

    Fix: from core.diagnostics import emit   ← always, no alias
    """
    violations = []
    for path in files:
        rel = _rel(path)
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "core.diagnostics":
                continue
            for alias in node.names:
                if alias.name == "emit" and alias.asname is not None and alias.asname != "emit":
                    violations.append(
                        f"  {rel}:{node.lineno} — 'emit as {alias.asname}' alias hides"
                        f" emit() calls from check 9; fix: from core.diagnostics import emit"
                    )
    return violations


def check_star_imports(files: list[Path]) -> list[str]:
    """Violation if any module uses 'from X import *' (explicit > implicit)."""
    violations = []
    for path in files:
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        src = node.module or "?"
                        violations.append(
                            f"  {_rel(path)}:{node.lineno} — "
                            f"star import 'from {src} import *' (explicit > implicit)"
                        )
    return violations


def _is_allowed_databank_import(module_name: str, names: list[str] | None = None) -> bool:
    """Allow only Treasurer door imports from databank/ across module boundaries."""
    if module_name == "databank.treasurer":
        return True
    if module_name == "databank" and names:
        return set(names) <= {"treasurer", "Treasurer"}
    return False


def _is_allowed_cross_module_import(
    importer_top: str,
    module_name: str,
    imported_names: list[str] | None = None,
) -> bool:
    target_top = module_name.split(".")[0]
    if target_top == importer_top:
        return True
    if target_top == "core":
        return True
    if target_top == "contracts":
        return True
    if target_top == "databank":
        return _is_allowed_databank_import(module_name, imported_names)
    return (importer_top, target_top) in _ALLOWED_IMPORT_PAIRS


def check_cross_module_import_allowlist(files: list[Path]) -> list[str]:
    """Violation if a cross-module import is not explicitly allowlisted."""
    violations = []
    for path in files:
        parts = path.relative_to(_PROJECT_ROOT).parts
        if not parts:
            continue
        # Exempt: .dev/ test scripts and root __init__.py (ComfyUI loader glue)
        if parts[0] == ".dev":
            continue
        if parts == ("__init__.py",):
            continue

        my_top = parts[0]
        if my_top not in _OWN_MODULES:
            continue
        tree = _parse(path)
        if tree is None:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    target_top = module_name.split(".")[0]
                    if target_top not in _OWN_MODULES:
                        continue
                    if _is_allowed_cross_module_import(my_top, module_name):
                        continue
                    violations.append(
                        f"  {_rel(path)}:{node.lineno} — "
                        f"{my_top}/ imports {module_name}, but {target_top}/ is not an allowed boundary door"
                    )
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if not module_name:
                    continue
                target_top = module_name.split(".")[0]
                if target_top not in _OWN_MODULES:
                    continue
                imported_names = [alias.name for alias in node.names]
                if _is_allowed_cross_module_import(my_top, module_name, imported_names):
                    continue
                violations.append(
                    f"  {_rel(path)}:{node.lineno} — "
                    f"{my_top}/ imports from {module_name}, but {target_top}/ is not an allowed boundary door"
                )
    return violations


def check_private_cross_module_imports(files: list[Path]) -> list[str]:
    """Violation if a module imports a private symbol (_name) from another top-level module.

    Boundary Door Law: no module may reach into another module's internals.
    Private symbols (prefixed with _) are internal contracts — importing them
    across module boundaries couples modules to implementation details.

    .dev/ scripts are exempt (they legitimately inspect internals for testing).
    """
    violations = []
    for path in files:
        parts = path.relative_to(_PROJECT_ROOT).parts
        if not parts:
            continue
        # .dev/ test scripts are exempt
        if parts[0] == ".dev":
            continue

        my_top = parts[0]
        tree = _parse(path)
        if tree is None:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            source_top = node.module.split(".")[0]
            # Only flag cross-module imports between known first-party modules
            if source_top == my_top or source_top not in _OWN_MODULES:
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(
                        f"  {_rel(path)}:{node.lineno} — "
                        f"imports private symbol '{source_top}.{alias.name}' "
                        f"across module boundary (Boundary Door Law)"
                    )
    return violations


def check_arch_cheat_cleanup(files: list[Path]) -> list[str]:
    """Violation if an ARCH.CHEAT emit() call is missing a 'cleanup=' keyword argument.

    The Architectural Cheat Ledger (Constitution §Cheat Ledger) requires every
    intentional rule-break to document its own removal condition via cleanup='...'
    in the emit() context kwargs.
    """
    violations = []
    for path in files:
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_emit = (isinstance(func, ast.Name) and func.id == "emit") or (
                isinstance(func, ast.Attribute) and func.attr == "emit"
            )
            if not is_emit:
                continue
            # The code string must be "ARCH.CHEAT" as the second positional arg
            if len(node.args) < 2:
                continue
            code_node = node.args[1]
            if not (isinstance(code_node, ast.Constant) and code_node.value == "ARCH.CHEAT"):
                continue
            # Check for cleanup= keyword argument
            has_cleanup = any(kw.arg == "cleanup" for kw in node.keywords)
            if not has_cleanup:
                violations.append(
                    f"  {_rel(path)}:{node.lineno} — "
                    f"ARCH.CHEAT emit() missing 'cleanup=...' context kwarg (Cheat Ledger rule)"
                )
    return violations



def _import_healthcheck_module(module_name: str):
    """Import one project module for integrity checks, returning (module, error)."""
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # pragma: no cover - failure path only
        return None, f"{module_name}: import failed ({exc.__class__.__name__}: {exc})"


def _check_frontend_hero_roster_contract() -> str | None:
    """Verify the frontend expects a payload-fed hero roster contract."""
    types_file = _PROJECT_ROOT / "operator_app" / "frontend" / "src" / "workbench" / "types.ts"
    if not types_file.exists():
        return f"{_rel(types_file)}: file not found"

    types_source = types_file.read_text(encoding="utf-8")
    if "hero_roster: HeroMetricDescriptor[];" not in types_source:
        return f"{_rel(types_file)}: ReviewSummaryPayload hero_roster field not found"

    use_review_data_file = _PROJECT_ROOT / "operator_app" / "frontend" / "src" / "workbench" / "useReviewData.ts"
    if not use_review_data_file.exists():
        return f"{_rel(use_review_data_file)}: file not found"

    use_review_data_source = use_review_data_file.read_text(encoding="utf-8")
    if "setHeroRoster(payload.hero_roster ?? []);" not in use_review_data_source:
        return f"{_rel(use_review_data_file)}: payload hero_roster is not threaded into frontend state"
    return None


def check_diagnostic_codes(files: list[Path]) -> list[str]:
    """Violation if an emit() code string is not in contracts/diagnostic_codes.py.

    Fails hard if the registry file is missing — the file is now required.
    """
    registry = _load_diagnostic_codes()
    if registry is None:
        return ["  contracts/diagnostic_codes.py: file missing — all emit() codes are unvalidated"]

    violations = []
    for path in files:
        tree = _parse(path)
        if tree is None:
            continue
        for code, lineno in _emit_calls_in(tree):
            if code not in registry:
                violations.append(
                    f"  {_rel(path)}:{lineno} — emit() code '{code}' not in contracts/diagnostic_codes.py"
                )
    return violations


def check_contracts_integrity() -> list[str]:
    """Validate cross-links between contracts/ metadata surfaces and operator_app usage."""
    violations: list[str] = []

    metrics_mod, metrics_err = _import_healthcheck_module("contracts.metrics_registry")
    metric_labels_mod, metric_labels_err = _import_healthcheck_module("contracts.metric_labels")
    procedures_mod, procedures_err = _import_healthcheck_module("contracts.procedures_registry")
    recipe_mod, recipe_err = _import_healthcheck_module("contracts.recipe")
    review_payload_mod, review_payload_err = _import_healthcheck_module("command_center.review_payload")

    for err in (metrics_err, metric_labels_err, procedures_err, recipe_err, review_payload_err):
        if err is not None:
            violations.append(f"  contracts integrity — {err}")
    if violations:
        return violations

    metric_definitions = list(getattr(metrics_mod, "_METRIC_DEFINITIONS", []))
    metrics_registry = dict(getattr(metrics_mod, "METRICS_REGISTRY", {}))
    metric_labels = dict(getattr(metric_labels_mod, "METRIC_LABELS", {}))
    procedures_registry = dict(getattr(procedures_mod, "PROCEDURES_REGISTRY", {}))
    featured_metrics = list(getattr(recipe_mod, "FEATURED_HERO_METRICS", []))
    build_metric_metadata = getattr(review_payload_mod, "build_metric_metadata")
    build_hero_roster = getattr(review_payload_mod, "build_hero_roster")
    frontend_hero_roster_err = _check_frontend_hero_roster_contract()
    if frontend_hero_roster_err is not None:
        violations.append(f"  contracts integrity — {frontend_hero_roster_err}")
        return violations

    metric_keys_seen: set[str] = set()
    for index, entry in enumerate(metric_definitions):
        key = entry.get("key")
        if not isinstance(key, str) or not key:
            violations.append(
                f"  contracts/metrics_registry.py: definition #{index} missing non-empty 'key'"
            )
            continue
        if key in metric_keys_seen:
            violations.append(
                f"  contracts/metrics_registry.py: duplicate metric key '{key}' in _METRIC_DEFINITIONS"
            )
        metric_keys_seen.add(key)

    for registry_key, entry in metrics_registry.items():
        if entry.get("key") != registry_key:
            violations.append(
                f"  contracts/metrics_registry.py: registry key '{registry_key}' mismatches entry key '{entry.get('key')}'"
            )

        for rel_field in ("reliability_metric_key", "dropped_metric_key"):
            rel_key = entry.get(rel_field)
            if rel_key is None:
                continue
            if rel_key not in metrics_registry:
                violations.append(
                    f"  contracts/metrics_registry.py: metric '{registry_key}' references missing {rel_field} '{rel_key}'"
                )
            elif metrics_registry[rel_key].get("deprecated"):
                violations.append(
                    f"  contracts/metrics_registry.py: metric '{registry_key}' references deprecated {rel_field} '{rel_key}'"
                )

        def _check_related_metric_list(field_name: str, relation_label: str) -> list[str]:
            rel_keys = entry.get(field_name)
            if rel_keys is None:
                return []
            if not isinstance(rel_keys, list):
                violations.append(
                    f"  contracts/metrics_registry.py: metric '{registry_key}' has non-list {field_name}"
                )
                return []
            for rel_key in rel_keys:
                if rel_key not in metrics_registry:
                    violations.append(
                        f"  contracts/metrics_registry.py: metric '{registry_key}' references missing {relation_label} '{rel_key}'"
                    )
                elif metrics_registry[rel_key].get("deprecated"):
                    violations.append(
                        f"  contracts/metrics_registry.py: metric '{registry_key}' references deprecated {relation_label} '{rel_key}'"
                    )
            return rel_keys

        component_keys = _check_related_metric_list("component_metric_keys", "component metric")
        _check_related_metric_list("peer_metric_keys", "peer metric")
        _check_related_metric_list("selection_metric_keys", "selection metric")
        _check_related_metric_list("inspection_graph_metric_keys", "inspection graph metric")
        if component_keys and "inspection_graph_metric_keys" not in entry:
            violations.append(
                f"  contracts/metrics_registry.py: metric '{registry_key}' has component_metric_keys but no inspection_graph_metric_keys"
            )

        if registry_key not in metric_labels:
            violations.append(
                f"  contracts/metric_labels.py: missing METRIC_LABELS entry for metric '{registry_key}'"
            )

    procedure_keys_seen: set[str] = set()
    for registry_key, spec in procedures_registry.items():
        spec_key = getattr(spec, "key", None)
        if spec_key in procedure_keys_seen:
            violations.append(
                f"  contracts/procedures_registry.py: duplicate procedure key '{spec_key}'"
            )
        if spec_key != registry_key:
            violations.append(
                f"  contracts/procedures_registry.py: registry key '{registry_key}' mismatches ProcedureSpec.key '{spec_key}'"
            )
        if isinstance(spec_key, str):
            procedure_keys_seen.add(spec_key)

        for metric_key in getattr(spec, "applies_to", ()):
            if metric_key not in metrics_registry:
                violations.append(
                    f"  contracts/procedures_registry.py: procedure '{registry_key}' applies_to missing metric '{metric_key}'"
                )
            elif metrics_registry[metric_key].get("deprecated"):
                violations.append(
                    f"  contracts/procedures_registry.py: procedure '{registry_key}' applies_to deprecated metric '{metric_key}'"
                )

    featured_keys_seen: set[str] = set()
    for featured in featured_metrics:
        featured_key = getattr(featured, "key", None)
        metric_key = getattr(featured, "metric_key", None)
        if featured_key in featured_keys_seen:
            violations.append(
                f"  contracts/recipe.py: duplicate featured hero key '{featured_key}'"
            )
        if isinstance(featured_key, str):
            featured_keys_seen.add(featured_key)

        if metric_key not in metrics_registry:
            violations.append(
                f"  contracts/recipe.py: featured metric '{featured_key}' references missing metric '{metric_key}'"
            )
            continue
        metric_entry = metrics_registry[metric_key]
        if metric_entry.get("deprecated"):
            violations.append(
                f"  contracts/recipe.py: featured metric '{featured_key}' promotes deprecated metric '{metric_key}'"
            )
        # If a promoted metric declares hero slots, they must resolve cleanly.
        for rel_field in ("reliability_metric_key", "dropped_metric_key"):
            rel_key = metric_entry.get(rel_field)
            if rel_key is None:
                continue
            if rel_key not in metrics_registry:
                violations.append(
                    f"  contracts/recipe.py: featured metric '{featured_key}' declares missing {rel_field} '{rel_key}'"
                )
            elif metrics_registry[rel_key].get("deprecated"):
                violations.append(
                    f"  contracts/recipe.py: featured metric '{featured_key}' declares deprecated {rel_field} '{rel_key}'"
                )

    recipe_hero_pairs = [
        (getattr(featured, "key", None), getattr(featured, "label", None))
        for featured in featured_metrics
    ]
    payload_hero_pairs = [
        (item.get("key"), item.get("label"))
        for item in build_hero_roster(build_metric_metadata())
    ]
    if payload_hero_pairs != recipe_hero_pairs:
        violations.append(
            "  operator_app/backend/review_payload.py: build_hero_roster does not match contracts/recipe.py FEATURED_HERO_METRICS"
        )

    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    files = _all_py_files()
    total_violations = 0
    all_clean = True

    sections: list[tuple[str, list[str]]] = []

    sections.append(("DB driver imports outside databank/",
                      check_db_drivers(files)))

    sections.append(("ComfyUI imports outside extractor/ comfyui/ lab/",
                      check_comfyui_imports(files)))

    sections.append(("Hard-coded absolute paths outside core/paths.py",
                      check_hardcoded_paths(files)))

    sections.append(("print() calls outside .dev/, lab/, or core/diagnostics.py",
                      check_print_statements(files)))

    sections.append(("logging module usage (use emit() from core.diagnostics instead)",
                      check_logging_module_usage(files)))

    sections.append(("Star imports (from X import *)",
                      check_star_imports(files)))

    sections.append(("Module boundary violations (explicit allowlist)",
                      check_cross_module_import_allowlist(files)))

    sections.append(("Private symbol imports across module boundaries (Boundary Door Law)",
                      check_private_cross_module_imports(files)))

    sections.append(("ARCH.CHEAT emit() missing cleanup= kwarg",
                      check_arch_cheat_cleanup(files)))

    sections.append(("Diagnostic code registry — all emit() codes registered",
                      check_diagnostic_codes(files)))

    sections.append(("Contracts / review metadata integrity",
                      check_contracts_integrity()))

    sections.append(("Aliased emit imports (hides calls from code registry check)",
                      check_aliased_emit_imports(files)))

    # Report
    for title, violations in sections:
        if violations:
            all_clean = False
            total_violations += len(violations)
            print(f"\n[FAIL] {title} ({len(violations)} violation(s)):")
            for v in violations:
                print(v)
        else:
            status = "SKIP" if "deferred" in title else "OK"
            print(f"[{status}]  {title}")

    print()
    if all_clean:
        print("dev_healthcheck: PASS — 0 violations")
        return 0
    else:
        print(f"dev_healthcheck: FAIL — {total_violations} violation(s) found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
