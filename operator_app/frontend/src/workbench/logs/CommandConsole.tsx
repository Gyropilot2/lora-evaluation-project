import type { FormEvent } from "react";

type CommandConsoleProps = {
  busy: boolean;
  commandText: string;
  commandHint: string;
  hasEntries: boolean;
  onCommandTextChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onClearLog: () => void;
};

export function CommandConsole({ busy, commandText, commandHint, hasEntries, onCommandTextChange, onSubmit, onClearLog }: CommandConsoleProps) {
  return (
    <section className="panel logs-console-panel">
      <form className="logs-console-form" onSubmit={onSubmit}>
        <label className="logs-console-label" htmlFor="logs-command-input">Command</label>
        <input id="logs-command-input" className="logs-console-input" value={commandText} onChange={(event) => onCommandTextChange(event.target.value)} placeholder={commandHint} autoComplete="off" spellCheck={false} />
        <div className="logs-console-actions">
          <button type="submit" disabled={busy || commandText.trim().length === 0}>{busy ? "Running…" : "Run"}</button>
          <button type="button" className="is-ghost" onClick={onClearLog} disabled={busy || !hasEntries}>Clear log</button>
        </div>
      </form>
    </section>
  );
}
