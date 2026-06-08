// ─── Pattern definitions ───────────────────────────────────────────────
// Keys must match what the model returns as `final_label` (first 2 chars
// of training filenames, lowercased).

export type StrumStep = "D" | "U" | "X" | "-" | "u";
//   D = downstroke ↓     U = upstroke ↑
//   X = muted/dead strum ×     - = empty (skip)

export type ArpStep = {
  strings: number[];   // 1 = highest e, 6 = lowest E. Empty array = rest.
  isBass?: boolean;
};

export type Pattern =
  | { kind: "strum"; label: string; name: string; description: string; color: string; steps: StrumStep[] }
  | { kind: "arp";   label: string; name: string; description: string; color: string; steps: ArpStep[] };

export const PATTERNS: Record<string, Pattern> = {
  "6b": {
    kind: "strum",
    label: "6b",
    name: "Бій",
    description: "D - D U - U D U",
    color: "#facc15",   // yellow
    steps: ["D", "-", "D", "U", "-", "U", "D", "U"],
  },
  "6x": {
    kind: "strum",
    label: "6x",
    name: "Бій (із заглушкою)",
    description: "D - X U - U X U",
    color: "#f97316",   // orange
    steps: ["D", "-", "X", "U", "-", "U", "X", "U"],
  },
  "8b": {
    kind: "strum",
    label: "8b",
    name: "Бій",
    description: "D D U U U D U D",
    color: "#ec4899",   // pink
    steps: ["D", "D", "U", "U", "U", "D", "U", "D"],
  },
  "6a": {
    kind: "arp",
    label: "6a",
    name: "Арпеджіо 1",
    description: "b 3 2 1 2 3",
    color: "#10b981",   // emerald
    steps: [
      { isBass: true, strings: [6] },
      { strings: [3] },
      { strings: [2] },
      { strings: [1] },
      { strings: [2] },
      { strings: [3] },
    ],
  },
  "8a": {
    kind: "arp",
    label: "8a",
    name: "Арпеджіо 2",
    description: "b 3 2 3 1 3 2 3",
    color: "#06b6d4",   // cyan
    steps: [
      { isBass: true, strings: [6] },
      { strings: [3] },
      { strings: [2] },
      { strings: [3] },
      { strings: [1] },
      { strings: [3] },
      { strings: [2] },
      { strings: [3] },
    ],
  },
  "wa": {
    kind: "arp",
    label: "wa",
    name: "Арпеджіо з акордом",
    description: "b 2+3+4 2+3+4",
    color: "#3b82f6",   // blue
    steps: [
      { isBass: true, strings: [6] },
      { strings: [2, 3, 4] },
      { strings: [2, 3, 4] },
    ],
  },
};

export const PATTERN_ORDER = ["8a", "8b", "wa", "6a", "6b", "6x"];

// Color for a class label — falls back to neutral grey if unknown.
export function colorFor(label: string): string {
  return PATTERNS[label]?.color ?? "#64748b";
}
