import "./CompareTray.css";
import type { FocusKind } from "../types";

type CompareChip = {
  id: string;
  label: string;
  isFocused?: boolean;
};

type CompareTrayProps = {
  level: FocusKind | null;
  chips: CompareChip[];
  onClear: () => void;
  onFocusChip: (level: FocusKind, id: string) => void;
  onRemoveChip: (level: FocusKind, id: string) => void;
};

export function CompareTray({ level, chips, onClear, onFocusChip, onRemoveChip }: CompareTrayProps) {
  if (!level || chips.length === 0) return null;

  return (
    <section className="compare-tray">
      <div className="compare-tray-header">
        <span>
          {chips.length} {level} items selected
        </span>
        <button type="button" onClick={onClear}>
          clear
        </button>
      </div>
      <div className="compare-chip-list">
        {chips.map((chip) => (
          <button
            key={chip.id}
            type="button"
            className={`compare-chip${chip.isFocused ? " is-focused" : ""}`}
            onClick={() => onFocusChip(level, chip.id)}
          >
            <span>{chip.label}</span>
            <span
              onClick={(event) => {
                event.stopPropagation();
                onRemoveChip(level, chip.id);
              }}
            >
              x
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
