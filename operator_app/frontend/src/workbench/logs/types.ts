export type CommandKind =
  | "help"
  | "summary"
  | "list-loras"
  | "list-evals"
  | "health"
  | "errors"
  | "write-review-dump"
  | "run-batch"
  | "list-workflows"
  | "onboard-workflow"
  | "get-method"
  | "get-eval"
  | "get-sample";

export type CommandSurface = "command-center";

export type PresetMode = "run" | "prefill";

export type CommandPreset = {
  label: string;
  command: string;
  description: string;
  mode: PresetMode;
};

export type PresetGroup = {
  title: string;
  surface: CommandSurface;
  items: CommandPreset[];
};

export type ParsedCommand =
  | {
      kind: CommandKind;
      mode: "remote";
      surface: CommandSurface;
      url: string;
      httpMethod: "GET" | "POST";
    }
  | {
      kind: "help";
      mode: "local";
      surface: "command-center";
    };

export type LogEntry = {
  id: string;
  command: string;
  createdAt: string;
  kind: CommandKind;
  status: "running" | "success" | "error";
  result?: unknown;
  error?: string;
};
