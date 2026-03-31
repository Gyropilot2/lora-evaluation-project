export type CopyFieldRow = {
  label: string;
  value: string;
};

export type CopySection = {
  title: string;
  rows: CopyFieldRow[];
};

export async function copyPlainText(text: string): Promise<void> {
  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (!normalized) return;

  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(normalized);
      return;
    } catch {
      // Fall through to the manual copy path if clipboard permissions fail.
    }
  }

  if (typeof document === "undefined") {
    throw new Error("clipboard_unavailable");
  }

  const textarea = document.createElement("textarea");
  textarea.value = normalized;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";
  textarea.style.inset = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();

  const copied = document.execCommand("copy");
  document.body.removeChild(textarea);

  if (!copied) {
    throw new Error("clipboard_unavailable");
  }
}

export function formatCopySections(args: {
  title?: string | null;
  subtitle?: string | null;
  sections: CopySection[];
}): string {
  const lines: string[] = [];
  const title = args.title?.trim();
  const subtitle = args.subtitle?.trim();

  if (title) lines.push(title);
  if (subtitle && subtitle !== title) lines.push(subtitle);

  for (const section of args.sections) {
    const rows = section.rows.filter((row) => row.label.trim() || row.value.trim());
    if (rows.length === 0) continue;
    if (lines.length > 0) lines.push("");
    lines.push(section.title);
    for (const row of rows) {
      lines.push(`${row.label}\t${row.value}`);
    }
  }

  return lines.join("\n");
}
