import { CopyButton } from "./CopyButton";
import { MetricLabelWithHelp } from "./MetricLabelWithHelp";

type RailKvRowProps = {
  label: string;
  value: string;
  description?: string | null;
  copyableValue?: boolean;
  className?: string;
};

function isTokenLabel(label: string): boolean {
  return /(?:id|hash)$/i.test(label.trim());
}

export function RailKvRow({
  label,
  value,
  description,
  copyableValue = false,
  className = "",
}: RailKvRowProps) {
  const token = isTokenLabel(label);
  const rowClassName = `rail-kv-row${token ? " rail-kv-row--token" : ""}${className ? ` ${className}` : ""}`;
  const valueClassName = `rail-kv-value${token ? " rail-kv-value--token" : ""}`;

  return (
    <div className={rowClassName}>
      <MetricLabelWithHelp label={label} description={description} className="rail-kv-label" />
      <div className={valueClassName}>
        {copyableValue ? (
          <CopyButton
            text={value}
            label={value}
            copiedLabel={value}
            className={`copy-inline-value${token ? " rail-kv-value--token" : ""}`}
            title={`Copy ${label}`}
          />
        ) : (
          value
        )}
      </div>
    </div>
  );
}
