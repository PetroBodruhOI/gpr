import { PredictResult } from "../api/client";
import { PATTERNS } from "../data/patterns";
import PatternDisplay, { PatternLegend } from "./PatternDisplay";
import ChunkDistributionChart from "./ChunkDistributionChart";

interface Props {
  result: PredictResult;
}

const HIGH_CONFIDENCE_THRESHOLD = 0.75;     // 75%
const MAX_ALT_RECOMMENDATIONS = 3;

function pickRecommendations(
  probs: Record<string, number>,
): { label: string; prob: number }[] {
  const sorted = Object.entries(probs)
    .map(([label, prob]) => ({ label, prob }))
    .sort((a, b) => b.prob - a.prob);
  if (sorted.length === 0) return [];

  // High confidence → just the top class.
  if (sorted[0].prob >= HIGH_CONFIDENCE_THRESHOLD) {
    return [sorted[0]];
  }

  // Low confidence → all classes above chance level (1 / n_classes),
  // capped at MAX_ALT_RECOMMENDATIONS.
  const chance = 1 / sorted.length;
  return sorted
    .filter((s) => s.prob > chance)
    .slice(0, MAX_ALT_RECOMMENDATIONS);
}

function PatternCard({
  label,
  prob,
  emphasis,
}: {
  label: string;
  prob: number;
  emphasis: "primary" | "secondary";
}) {
  const pattern = PATTERNS[label];
  const pct = prob * 100;
  const isPrimary = emphasis === "primary";

  return (
    <div
      className={isPrimary ? "card-glow" : "card"}
      style={isPrimary ? undefined : { background: "rgba(10, 20, 36, 0.5)" }}
    >
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 mb-3">
        <div className={isPrimary ? "text-3xl md:text-4xl font-bold text-white" : "text-2xl font-bold text-white"}>
          {pattern ? pattern.name : label}
        </div>
        {pattern && (
          <span className="text-sm text-yellow-300/70 font-mono">{pattern.label}</span>
        )}
        <span className="ml-auto flex items-baseline gap-1">
          <span className="text-xl font-semibold text-yellow-300">{pct.toFixed(1)}%</span>
          <span className="text-white/50 text-sm">relevance</span>
        </span>
      </div>

      {pattern && (
        <p className="text-white/50 text-xs font-mono mb-4">{pattern.description}</p>
      )}

      {pattern ? (
        <>
          <div className="py-2 overflow-x-auto">
            <PatternDisplay pattern={pattern} size={isPrimary ? "lg" : "md"} />
          </div>
          <div className="mt-3">
            <PatternLegend kind={pattern.kind === "strum" ? "strum" : "arp"} />
          </div>
        </>
      ) : (
        <p className="text-white/40 text-sm italic">
          Pattern definition not found for "{label}"
        </p>
      )}
    </div>
  );
}

export default function ResultCard({ result }: Props) {
  const recommendations = pickRecommendations(result.mean_probs);
  const isHighConf = result.final_conf >= HIGH_CONFIDENCE_THRESHOLD;

  return (
    <div className="space-y-6 mt-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-11 h-11 rounded-lg bg-emerald-500/15 flex items-center justify-center border border-emerald-500/30">
            <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-white">
              {isHighConf ? "Recommended Pattern" : "Possible Patterns"}
            </h2>
            <p className="text-white/50 text-sm">
              {isHighConf
                ? "Strong match for your track"
                : `Model is uncertain — showing top ${recommendations.length} candidates`}
            </p>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="space-y-4">
        {recommendations.map((r, idx) => (
          <PatternCard
            key={r.label}
            label={r.label}
            prob={r.prob}
            emphasis={idx === 0 ? "primary" : "secondary"}
          />
        ))}
      </div>

      {/* Chunk distribution chart */}
      <ChunkDistributionChart chunks={result.chunks} />

      {/* Timeline table */}
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-5 flex items-center gap-2">
          <span className="w-1 h-5 bg-emerald-500 rounded-full" />
          Timeline
          <span className="text-white/40 text-sm font-normal ml-2">
            {result.n_chunks} segments
          </span>
        </h3>
        <div className="overflow-x-auto rounded-lg border border-white/5">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-900/40 border-b border-white/5">
                <th className="px-4 py-3 text-left font-medium text-white/50 uppercase text-xs tracking-wider">#</th>
                <th className="px-4 py-3 text-left font-medium text-white/50 uppercase text-xs tracking-wider">Time</th>
                <th className="px-4 py-3 text-left font-medium text-white/50 uppercase text-xs tracking-wider">Pattern</th>
                <th className="px-4 py-3 text-left font-medium text-white/50 uppercase text-xs tracking-wider">Conf</th>
              </tr>
            </thead>
            <tbody>
              {result.chunks.map((c) => (
                <tr
                  key={c.chunk_idx}
                  className="border-b border-white/5 hover:bg-white/[0.02] transition-colors"
                >
                  <td className="px-4 py-3 text-white/40 font-mono">{c.chunk_idx}</td>
                  <td className="px-4 py-3 text-white/70 font-mono">
                    {c.time_start.toFixed(1)}s – {c.time_end.toFixed(1)}s
                  </td>
                  <td className="px-4 py-3 font-medium text-white">{c.label}</td>
                  <td className="px-4 py-3">
                    <span className="inline-block px-2.5 py-1 rounded-md text-xs font-semibold"
                          style={{
                            background: "rgba(234, 179, 8, 0.12)",
                            color: "#fde047",
                            border: "1px solid rgba(234, 179, 8, 0.25)"
                          }}>
                      {(c.confidence * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
