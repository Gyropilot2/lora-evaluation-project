import { isObjectRecord } from "./renderers";
import type { ParsedCommand } from "./types";

export async function executeParsedCommand(parsed: ParsedCommand): Promise<unknown> {
  if (parsed.mode === "local") {
    return { help: true };
  }

  const response = await fetch(parsed.url, {
    method: parsed.httpMethod,
  });
  const payload = (await response.json()) as unknown;
  if (!response.ok) {
    const detail =
      isObjectRecord(payload) && typeof payload.detail === "string"
        ? payload.detail
        : `Command failed (${response.status}).`;
    throw new Error(detail);
  }
  return payload;
}
