import { useEffect, useState } from "react";

import { copyPlainText } from "../clipboard";

type CopyButtonProps = {
  text: string;
  label?: string;
  copiedLabel?: string;
  className?: string;
  title?: string;
};

export function CopyButton({
  text,
  label = "Copy",
  copiedLabel = "Copied",
  className = "",
  title,
}: CopyButtonProps) {
  const [state, setState] = useState<"idle" | "copied">("idle");
  const canCopy = text.trim().length > 0;

  useEffect(() => {
    if (state !== "copied") return undefined;
    const timeout = window.setTimeout(() => setState("idle"), 1200);
    return () => window.clearTimeout(timeout);
  }, [state]);

  async function handleClick() {
    if (!canCopy) return;
    try {
      await copyPlainText(text);
      setState("copied");
    } catch {
      setState("idle");
    }
  }

  const buttonClassName = `copy-button${state === "copied" ? " is-copied" : ""}${className ? ` ${className}` : ""}`;

  return (
    <button
      type="button"
      className={buttonClassName}
      onClick={() => void handleClick()}
      disabled={!canCopy}
      title={title ?? label}
    >
      {state === "copied" ? copiedLabel : label}
    </button>
  );
}
