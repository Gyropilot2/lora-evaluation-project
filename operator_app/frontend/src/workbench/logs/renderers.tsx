import { HELP_SECTIONS } from "./manifest";
import type { LogEntry } from "./types";

export function shortTimestamp(iso: string) {
  const date = new Date(iso);
  return Number.isNaN(date.valueOf()) ? iso : date.toLocaleTimeString();
}

export function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function displayValue(value: unknown) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function renderArrayTable(rows: unknown[]) {
  if (rows.length === 0 || !rows.every((row) => isObjectRecord(row))) return <pre className="logs-json">{JSON.stringify(rows, null, 2)}</pre>;
  const keySet = new Set<string>();
  for (const row of rows.slice(0, 10)) {
    for (const [key, value] of Object.entries(row)) {
      if (typeof value === "object" && value !== null) continue;
      keySet.add(key);
      if (keySet.size >= 6) break;
    }
    if (keySet.size >= 6) break;
  }
  const keys = Array.from(keySet);
  if (keys.length === 0) return <pre className="logs-json">{JSON.stringify(rows, null, 2)}</pre>;
  return (
    <div className="logs-table-wrap">
      <table className="data-table">
        <thead><tr>{keys.map((key) => <th key={key}>{key}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${index}-${keys.map((key) => displayValue((row as Record<string, unknown>)[key])).join("|")}`}>
              {keys.map((key) => <td key={key}>{displayValue((row as Record<string, unknown>)[key])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function renderObjectBlock(value: Record<string, unknown>) {
  const flatEntries = Object.entries(value).filter(([, item]) => typeof item !== "object" || item === null);
  if (flatEntries.length === 0) return <pre className="logs-json">{JSON.stringify(value, null, 2)}</pre>;
  return (
    <dl className="logs-kv-list">
      {flatEntries.map(([key, item]) => (
        <div key={key} className="logs-kv-row">
          <dt>{key}</dt>
          <dd>{displayValue(item)}</dd>
        </div>
      ))}
    </dl>
  );
}

export function renderOutput(entry: LogEntry) {
  if (entry.status === "running") return <div className="summary-line">Running…</div>;
  if (entry.status === "error") return <div className="logs-error-copy">{entry.error}</div>;
  if (entry.kind === "help") {
    return (
      <div className="logs-help-block">
        {HELP_SECTIONS.map((section) => (
          <div key={section.title} className="logs-subsection">
            <div className="logs-subhead">{section.title}</div>
            <pre className="logs-json">{section.lines.join("\n")}</pre>
          </div>
        ))}
      </div>
    );
  }
  if (entry.kind === "summary" && isObjectRecord(entry.result)) {
    const counts = isObjectRecord(entry.result.counts) ? entry.result.counts : null;
    const recent = isObjectRecord(entry.result.recent) ? entry.result.recent : null;
    const samples = Array.isArray(recent?.samples) ? recent.samples : [];
    const evals = Array.isArray(recent?.evals) ? recent.evals : [];
    return <>{counts ? renderObjectBlock(counts) : null}<div className="logs-subsection"><div className="logs-subhead">Recent evals</div>{renderArrayTable(evals)}</div><div className="logs-subsection"><div className="logs-subhead">Recent samples</div>{renderArrayTable(samples)}</div></>;
  }
  if (entry.kind === "health" && isObjectRecord(entry.result)) {
    const db = isObjectRecord(entry.result.db) ? entry.result.db : null;
    const diagnostics = isObjectRecord(entry.result.diagnostics) ? entry.result.diagnostics : null;
    const scorecard = isObjectRecord(entry.result.scorecard) ? entry.result.scorecard : null;
    return <>{db ? <div className="logs-subsection"><div className="logs-subhead">DB snapshot</div>{renderObjectBlock(db)}</div> : null}{diagnostics ? <div className="logs-subsection"><div className="logs-subhead">Diagnostics</div>{renderObjectBlock(diagnostics)}</div> : null}{scorecard ? <details className="logs-detail-block"><summary>Full scorecard</summary><pre className="logs-json">{JSON.stringify(scorecard, null, 2)}</pre></details> : null}</>;
  }
  if (entry.kind === "write-review-dump" && isObjectRecord(entry.result)) {
    return <>{renderObjectBlock(entry.result)}{isObjectRecord(entry.result.summary) ? <div className="logs-subsection"><div className="logs-subhead">Dump summary</div>{renderObjectBlock(entry.result.summary)}</div> : null}</>;
  }
  if (entry.kind === "list-workflows" && isObjectRecord(entry.result)) {
    const ready = Array.isArray(entry.result.ready) ? entry.result.ready as string[] : [];
    const pending = Array.isArray(entry.result.pending_onboard) ? entry.result.pending_onboard as string[] : [];
    return (
      <>
        <div className="logs-subsection">
          <div className="logs-subhead">Ready — {ready.length} workflow{ready.length !== 1 ? "s" : ""}</div>
          {ready.length === 0
            ? <div className="summary-line">No workflows onboarded yet. Drop a raw export in staging and run onboard-workflow.</div>
            : <pre className="logs-json">{ready.join("\n")}</pre>}
        </div>
        <div className="logs-subsection">
          <div className="logs-subhead">Staging — {pending.length} pending onboard</div>
          {pending.length === 0
            ? <div className="summary-line">Staging is empty.</div>
            : <pre className="logs-json">{pending.join("\n")}</pre>}
        </div>
      </>
    );
  }
  if (entry.kind === "onboard-workflow" && isObjectRecord(entry.result)) {
    const ok = entry.result.ok;
    return (
      <>
        <dl className="logs-kv-list">
          <div className="logs-kv-row"><dt>status</dt><dd>{ok ? "✓ onboarded" : "✗ failed"}</dd></div>
          {typeof entry.result.workflow_name === "string" && <div className="logs-kv-row"><dt>name</dt><dd>{entry.result.workflow_name}</dd></div>}
          {typeof entry.result.workspace_path === "string" && <div className="logs-kv-row"><dt>written to</dt><dd>{entry.result.workspace_path}</dd></div>}
          {typeof entry.result.error === "string" && <div className="logs-kv-row"><dt>error</dt><dd>{entry.result.error}</dd></div>}
        </dl>
      </>
    );
  }
  if (entry.kind === "run-batch" && isObjectRecord(entry.result)) {
    const { ok, dry_run, output, exit_code } = entry.result;
    return <><dl className="logs-kv-list"><div className="logs-kv-row"><dt>status</dt><dd>{ok ? "ok" : "failed"}</dd></div><div className="logs-kv-row"><dt>mode</dt><dd>{dry_run ? "dry-run" : "queued"}</dd></div>{exit_code !== null && exit_code !== undefined ? <div className="logs-kv-row"><dt>exit_code</dt><dd>{String(exit_code)}</dd></div> : null}</dl>{typeof output === "string" && output.trim() ? <pre className="logs-json">{output}</pre> : null}</>;
  }
  if (Array.isArray(entry.result)) return renderArrayTable(entry.result);
  if (isObjectRecord(entry.result)) {
    return <>{renderObjectBlock(entry.result)}<details className="logs-detail-block"><summary>Raw JSON</summary><pre className="logs-json">{JSON.stringify(entry.result, null, 2)}</pre></details></>;
  }
  return <pre className="logs-json">{JSON.stringify(entry.result, null, 2)}</pre>;
}
