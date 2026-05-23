import { ChunkResult } from "../api/client";
import { colorFor, PATTERNS } from "../data/patterns";

interface Props {
  chunks: ChunkResult[];
}

export default function ChunkDistributionChart({ chunks }: Props) {
  // Collect every class that appears anywhere in the chunks.
  const allClasses = Array.from(
    new Set(chunks.flatMap((c) => Object.keys(c.probs)))
  ).sort((a, b) => {
    const aKnown = a in PATTERNS;
    const bKnown = b in PATTERNS;
    if (aKnown !== bKnown) return aKnown ? -1 : 1;
    return a.localeCompare(b);
  });

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-white mb-1 flex items-center gap-2">
        <span className="w-1 h-5 bg-emerald-500 rounded-full" />
        Розподіл за часом
      </h3>
      <p className="text-white/50 text-sm mb-5 ml-3">
        Ймовірності кожного класу в усіх часових сегментах
      </p>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mb-5">
        {allClasses.map((cls) => {
          const known = PATTERNS[cls];
          return (
            <div key={cls} className="flex items-center gap-1.5 text-xs">
              <span
                className="w-3 h-3 rounded-sm inline-block"
                style={{ background: colorFor(cls) }}
              />
              <span className="font-mono text-white/80">{cls}</span>
              {known && (
                <span className="text-white/40 hidden sm:inline">{known.name}</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Vertical stacked bars — no gap between columns */}
      <div className="relative">
        {/* % gridlines */}
        <div className="absolute inset-y-0 left-0 right-0 flex flex-col-reverse justify-between pointer-events-none">
          {[0, 25, 50, 75, 100].map((v) => (
            <div key={v} className="relative">
              <div className="border-t border-white/5" />
              <span className="absolute -top-2 -left-9 text-[10px] text-white/30 font-mono w-8 text-right">
                {v}%
              </span>
            </div>
          ))}
        </div>

        {/* Bars */}
        <div className="flex h-56 ml-9 rounded-md overflow-hidden border border-white/5">
          {chunks.map((c) => {
            const segments = allClasses
              .map((cls) => ({ cls, p: c.probs[cls] ?? 0 }))
              .filter((s) => s.p > 0);
            return (
              <div
                key={c.chunk_idx}
                className="flex-1 flex flex-col-reverse h-full relative group"
              >
                {segments.map(({ cls, p }) => (
                  <div
                    key={cls}
                    style={{ height: `${p * 100}%`, background: colorFor(cls) }}
                    className="relative transition-all hover:brightness-125 flex items-center justify-center overflow-hidden"
                    title={`${cls}: ${(p * 100).toFixed(1)}%`}
                  >
                    {p * 100 >= 18 && (
                      <span className="text-[10px] font-bold text-slate-900 font-mono">
                        {cls}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            );
          })}
        </div>

        {/* X-axis labels */}
        <div className="flex ml-9 mt-2">
          {chunks.map((c) => (
            <div key={c.chunk_idx} className="flex-1 text-center px-1">
              <div className="text-[11px] text-white/70 font-mono">#{c.chunk_idx}</div>
              <div className="text-[10px] text-white/40 font-mono whitespace-nowrap">
                {c.time_start.toFixed(1)}–{c.time_end.toFixed(1)}s
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
