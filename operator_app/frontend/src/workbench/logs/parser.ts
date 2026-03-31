import { COMMAND_HINT } from "./manifest";
import type { ParsedCommand } from "./types";

export function newEntryId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

export function parsePositiveInt(token: string | undefined, fallback: number) {
  if (!token) return fallback;
  const value = Number(token);
  if (!Number.isInteger(value) || value < 1) throw new Error(`Expected a positive integer, got "${token}".`);
  return value;
}

export function buildUrl(path: string, params?: Record<string, string | number | undefined>) {
  if (!params) return path;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const query = search.toString();
  return query ? `${path}?${query}` : path;
}

export function parseCommand(raw: string): ParsedCommand {
  const tokens = raw.trim().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) throw new Error(`Enter a command first. Example: ${COMMAND_HINT}`);

  const [name, ...rest] = tokens;
  switch (name.toLowerCase()) {
    case "help":
      return { kind: "help", mode: "local", surface: "command-center" };
    case "health":
      return { kind: "health", mode: "remote", surface: "command-center", url: buildUrl("/api/cc/health", { diagnostics_window: parsePositiveInt(rest[0], 100) }), httpMethod: "GET" };
    case "errors":
      return { kind: "errors", mode: "remote", surface: "command-center", url: buildUrl("/api/cc/errors", { limit: parsePositiveInt(rest[0], 10) }), httpMethod: "GET" };
    case "summary":
      return { kind: "summary", mode: "remote", surface: "command-center", url: buildUrl("/api/cc/summary", { limit: parsePositiveInt(rest[0], 10) }), httpMethod: "GET" };
    case "list-loras":
      return { kind: "list-loras", mode: "remote", surface: "command-center", url: buildUrl("/api/cc/list-loras", { limit: parsePositiveInt(rest[0], 100) }), httpMethod: "GET" };
    case "list-evals": {
      const options: Record<string, string | number | undefined> = { limit: 10 };
      for (const token of rest) {
        if (/^\d+$/.test(token)) options.limit = parsePositiveInt(token, 10);
        else if (token.startsWith("method=") || token.startsWith("method_hash=")) options.method_hash = token.split("=", 2)[1];
        else if (token.startsWith("lora=") || token.startsWith("lora_hash=")) options.lora_hash = token.split("=", 2)[1];
        else throw new Error(`Unknown list-evals argument "${token}".`);
      }
      return { kind: "list-evals", mode: "remote", surface: "command-center", url: buildUrl("/api/cc/evals", options), httpMethod: "GET" };
    }
    case "write-review-dump":
      return { kind: "write-review-dump", mode: "remote", surface: "command-center", url: "/api/cc/review-dump", httpMethod: "POST" };
    case "get-method":
      if (!rest[0]) throw new Error("get-method requires a method hash.");
      return { kind: "get-method", mode: "remote", surface: "command-center", url: `/api/cc/methods/${encodeURIComponent(rest[0])}`, httpMethod: "GET" };
    case "get-eval":
      if (!rest[0]) throw new Error("get-eval requires an eval hash.");
      return { kind: "get-eval", mode: "remote", surface: "command-center", url: `/api/cc/evals/${encodeURIComponent(rest[0])}`, httpMethod: "GET" };
    case "get-sample":
      if (!rest[0]) throw new Error("get-sample requires a sample hash.");
      return { kind: "get-sample", mode: "remote", surface: "command-center", url: `/api/cc/samples/${encodeURIComponent(rest[0])}`, httpMethod: "GET" };
    case "list-workflows":
      return { kind: "list-workflows", mode: "remote", surface: "command-center", url: "/api/cc/list-workflows", httpMethod: "GET" };
    case "onboard-workflow": {
      if (!rest[0]) throw new Error("onboard-workflow requires a filename (e.g. onboard-workflow my_workflow.json).");
      return { kind: "onboard-workflow", mode: "remote", surface: "command-center", url: buildUrl("/api/cc/onboard-workflow", { filename: rest[0] }), httpMethod: "POST" };
    }
    case "run-batch": {
      let dryRun = false;
      let force = false;
      let includeDirty = false;
      let limit: number | undefined;
      let i = 0;
      while (i < rest.length) {
        const t = rest[i];
        if (t === "--dry-run") { dryRun = true; i++; }
        else if (t === "--force") { force = true; i++; }
        else if (t === "--include-dirty") { includeDirty = true; i++; }
        else if (t === "--limit") { if (!rest[i + 1]) throw new Error("--limit requires a value."); limit = parsePositiveInt(rest[i + 1], 1); i += 2; }
        else throw new Error(`Unknown run-batch argument "${t}". Allowed: --dry-run --limit N --force --include-dirty`);
      }
      return { kind: "run-batch", mode: "remote", surface: "command-center", url: buildUrl("/api/cc/run-batch", { dry_run: dryRun ? "true" : "false", ...(force ? { force: "true" } : {}), ...(includeDirty ? { include_dirty: "true" } : {}), ...(limit !== undefined ? { limit } : {}) }), httpMethod: "POST" };
    }
    default:
      throw new Error(`Unknown command "${name}". Try "help".`);
  }
}
