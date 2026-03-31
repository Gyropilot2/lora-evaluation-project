import type { CommandPreset, PresetGroup } from "./types";

type PresetRailProps = {
  busy: boolean;
  presetGroups: PresetGroup[];
  onUsePreset: (preset: CommandPreset) => void;
};

export function PresetRail({ busy, presetGroups, onUsePreset }: PresetRailProps) {
  return (
    <aside className="logs-sidebar">
      <section className="panel logs-command-panel">
        <div className="panel-heading">
          <h2>Command presets</h2>
          <span className="panel-tag">cc + app actions</span>
        </div>
        <p className="compact-copy">
          Command Center stays the stable operator door. App actions sit beside it when they do not honestly belong in `/api/cc`.
        </p>
        <div className="logs-preset-groups">
          {presetGroups.map((group) => (
            <section key={group.title} className="logs-preset-group">
              <div className="logs-subhead">
                {group.title}
                <span className="logs-surface-tag">{group.surface === "command-center" ? "cc" : "app"}</span>
              </div>
              <div className="logs-preset-list">
                {group.items.map((preset) => (
                  <button key={`${group.title}-${preset.label}`} type="button" className="logs-preset-button" disabled={busy} onClick={() => onUsePreset(preset)}>
                    <span className="logs-preset-topline">
                      <strong>{preset.label}</strong>
                      <span className="logs-preset-mode">{preset.mode === "run" ? "run" : "fill"}</span>
                    </span>
                    <span className="logs-preset-command">{preset.command.trim() || preset.command}</span>
                    <span className="logs-preset-copy">{preset.description}</span>
                  </button>
                ))}
              </div>
            </section>
          ))}
        </div>
      </section>
    </aside>
  );
}
