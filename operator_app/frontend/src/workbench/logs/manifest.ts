import type { PresetGroup } from "./types";

export const COMMAND_HINT =
  "help | health | errors 25 | summary 20 | run-batch --dry-run | list-evals 20 | get-method <hash>";

export const HELP_SECTIONS: Array<{ title: string; lines: string[] }> = [
  {
    title: "Command Center",
    lines: [
      "health [diagnostics_window]",
      "errors [limit]",
      "summary [limit]",
      "list-loras [limit]",
      "list-evals [limit] [method=<method_hash>] [lora=<lora_hash>]",
      "write-review-dump",
      "run-batch [--dry-run] [--limit N] [--force] [--include-dirty]",
      "list-workflows",
      "onboard-workflow <filename>",
      "get-method <method_hash>",
      "get-eval <eval_hash>",
      "get-sample <sample_hash>",
    ],
  },
];

export const PRESET_GROUPS: PresetGroup[] = [
  {
    title: "Command Center quick run",
    surface: "command-center",
    items: [
      { label: "Health", command: "health", description: "DataBank scorecard plus recent diagnostics snapshot.", mode: "run" },
      { label: "Errors", command: "errors 25", description: "Recent WARN / ERROR / FATAL diagnostics.", mode: "run" },
      { label: "Summary", command: "summary 20", description: "Counts plus recent eval/sample overview.", mode: "run" },
      {
        label: "Review dump",
        command: "write-review-dump",
        description: "Write the shared review JSON through the real Command Center door.",
        mode: "run",
      },
      {
        label: "Batch replay",
        command: "run-batch --dry-run",
        description: "Preview the replay plan (backfill missing measurements into existing samples). Remove --dry-run to execute.",
        mode: "prefill",
      },
      { label: "Workflows", command: "list-workflows", description: "Workspace templates ready to run + staging files pending onboard.", mode: "run" },
      { label: "Onboard", command: "onboard-workflow ", description: "Onboard a raw workflow from staging. Type the filename after the command.", mode: "prefill" },
      { label: "LoRAs", command: "list-loras 50", description: "Inventory of known LoRAs and their coverage.", mode: "run" },
      { label: "Evals", command: "list-evals 25", description: "Recent evals without leaving the app.", mode: "run" },
    ],
  },
  {
    title: "Command Center lookup",
    surface: "command-center",
    items: [
      { label: "Method", command: "get-method ", description: "Prefill a method lookup command.", mode: "prefill" },
      { label: "Eval", command: "get-eval ", description: "Prefill an eval lookup command.", mode: "prefill" },
      { label: "Sample", command: "get-sample ", description: "Prefill a sample lookup command.", mode: "prefill" },
      { label: "Help", command: "help", description: "Show the small allowed command grammar.", mode: "run" },
    ],
  },
];
