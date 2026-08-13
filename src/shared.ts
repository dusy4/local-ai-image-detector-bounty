export const REQUIRED_THRESHOLD = 0.65;
export type Signal = { source: string; score: number; detail: string };
export type ExpertScore = { source: string; score: number; active: boolean; detail: string };
export type AnalysisResult = {
  score: number;
  label: "AI" | "real";
  modelScore: number;
  experts: ExpertScore[];
  signals: Signal[];
  elapsedMs: number;
};
export type Settings = { enabled: boolean; minSize: number };
export const DEFAULT_SETTINGS: Settings = { enabled: true, minSize: 160 };

export function fuseScores(modelScore: number, signals: Signal[], experts: ExpertScore[] = []): number {
  const community = experts.find(expert => expert.active && expert.source === "Community Forensics");
  const modern = experts.find(expert => expert.active && expert.source === "Modern ConvNeXt");
  const hasCalibratedPair = Boolean(community && modern);
  const calibrated = hasCalibratedPair
    ? sigmoid(0.36182409415728856 + 1.2 * logit(modelScore) + 0.2 * logit(community.score) + 0.2 * logit(modern.score))
    : modelScore;
  let complement = 1 - clamp(calibrated);
  for (const signal of [...signals, ...experts.filter(expert => expert.active && (!hasCalibratedPair || (expert !== community && expert !== modern)))]) {
    if (signal.score < REQUIRED_THRESHOLD) continue;
    complement *= 1 - clamp(signal.score);
  }
  return clamp(1 - complement);
}

function logit(value: number): number {
  const bounded = Math.max(1e-7, Math.min(1 - 1e-7, value));
  return Math.log(bounded / (1 - bounded));
}

function sigmoid(value: number): number {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

export function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}
