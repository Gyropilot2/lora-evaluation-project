import { useMemo, useState, type FormEvent } from "react";

import "./LogsWorkspace.css";
import { CommandConsole } from "../logs/CommandConsole";
import { executeParsedCommand } from "../logs/executor";
import { HistoryPanel } from "../logs/HistoryPanel";
import { COMMAND_HINT, PRESET_GROUPS } from "../logs/manifest";
import { newEntryId, parseCommand } from "../logs/parser";
import { PresetRail } from "../logs/PresetRail";
import type { CommandPreset, LogEntry, ParsedCommand } from "../logs/types";

export function LogsWorkspace() {
  const [commandText, setCommandText] = useState("");
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [busy, setBusy] = useState(false);

  const reversedEntries = useMemo(() => [...entries].reverse(), [entries]);

  async function runCommand(rawCommand: string) {
    const command = rawCommand.trim();
    if (!command) return;

    const entryId = newEntryId();
    const createdAt = new Date().toISOString();
    let parsed: ParsedCommand;
    try {
      parsed = parseCommand(command);
    } catch (error) {
      setEntries((current) => [
        ...current,
        {
          id: entryId,
          command,
          createdAt,
          kind: "help",
          status: "error",
          error: error instanceof Error ? error.message : "Unable to parse command.",
        },
      ]);
      return;
    }

    setEntries((current) => [
      ...current,
      {
        id: entryId,
        command,
        createdAt,
        kind: parsed.kind,
        status: "running",
      },
    ]);

    if (parsed.mode !== "local") setBusy(true);
    try {
      const payload = await executeParsedCommand(parsed);
      setEntries((current) =>
        current.map((entry) =>
          entry.id === entryId
            ? {
                ...entry,
                status: "success",
                result: payload,
              }
            : entry,
        ),
      );
      setCommandText("");
    } catch (error) {
      setEntries((current) =>
        current.map((entry) =>
          entry.id === entryId
            ? {
                ...entry,
                status: "error",
                error: error instanceof Error ? error.message : "Unknown command failure.",
              }
            : entry,
        ),
      );
    } finally {
      if (parsed.mode !== "local") setBusy(false);
    }
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void runCommand(commandText);
  }

  function usePreset(preset: CommandPreset) {
    if (busy) return;
    if (preset.mode === "run") {
      void runCommand(preset.command);
      return;
    }
    setCommandText(preset.command);
  }

  return (
    <div className="logs-workspace">
      <PresetRail busy={busy} presetGroups={PRESET_GROUPS} onUsePreset={usePreset} />

      <main className="logs-main">
        <HistoryPanel entries={reversedEntries} />
        <CommandConsole
          busy={busy}
          commandText={commandText}
          commandHint={COMMAND_HINT}
          hasEntries={entries.length > 0}
          onCommandTextChange={setCommandText}
          onSubmit={onSubmit}
          onClearLog={() => setEntries([])}
        />
      </main>
    </div>
  );
}
