import "./MetricLabelWithHelp.css";

type MetricLabelWithHelpProps = {
  label: string;
  description?: string | null;
  className?: string;
};

export function MetricLabelWithHelp({ label, description, className = "" }: MetricLabelWithHelpProps) {
  const hasDescription = Boolean(description?.trim());
  const combinedClassName = `metric-label-help${hasDescription ? " is-described" : ""}${className ? ` ${className}` : ""}`;
  return (
    <span className={combinedClassName} title={description ?? undefined}>
      {label}
    </span>
  );
}
