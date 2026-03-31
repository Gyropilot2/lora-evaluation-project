import { renderOutput, shortTimestamp } from "./renderers";
import type { LogEntry } from "./types";

type HistoryPanelProps = {
  entries: LogEntry[];
};

export function HistoryPanel({ entries }: HistoryPanelProps) {
  return (
    <section className="panel logs-log-panel">
      <div className="panel-heading">
        <h2>Operator log</h2>
        <span className="panel-tag">{entries.length} entries</span>
      </div>
      <p className="compact-copy">This room keeps a local command history for the current session. It is intentionally not a raw shell.</p>
      <div className="logs-history">
        {entries.length === 0 ? (
          <section className="logs-empty-state">
            <div className="logs-subhead">Nothing run yet</div>
            <p className="summary-line">Start with <code>health</code>, <code>errors 25</code>, or a lookup command from the left rail.</p>
          </section>
        ) : (
          entries.map((entry) => (
            <article key={entry.id} className={`logs-entry is-${entry.status}`}>
              <header className="logs-entry-header">
                <div>
                  <div className="logs-entry-command">{entry.command}</div>
                  <div className="logs-entry-meta">{shortTimestamp(entry.createdAt)}</div>
                </div>
                <span className={`logs-status logs-status-${entry.status}`}>{entry.status}</span>
              </header>
              <div className="logs-entry-body">{renderOutput(entry)}</div>
            </article>
          ))
        )}
      </div>
    </section>
  );
}
