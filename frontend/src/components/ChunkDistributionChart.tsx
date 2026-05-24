import { useState } from "react";
import { ChunkResult } from "../api/client";
import { colorFor, PATTERNS } from "../data/patterns";

interface Props {
  chunks: ChunkResult[];
}

interface HoverInfo {
  chunkIdx: number;
  timeStart: number;
  timeEnd: number;
  cls: string;
  prob: number;
}

export default function ChunkDistributionChart({ chunks }: Props) {
  const [hover, setHover] = useState<HoverInfo | null>(null);

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

      {/* Hover tooltip — fixed-height slot so layout doesn't jump */}
      <div className="h-12 mb-2 flex items-center">
        {hover ? (
          <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-slate-900/70 border border-white/10 text-sm">
            <span
              className="w-3 h-3 rounded-sm inline-block flex-shrink-0"
              style={{ background: colorFor(hover.cls) }}
            />
            <span className="font-mono text-white/90">{hover.cls}</span>
            {PATTERNS[hover.cls] && (
              <span className="text-white/60 text-xs">
                {PATTERNS[hover.cls].name}
              </span>
            )}
            <span className="text-white/40 text-xs ml-2">
              сегмент #{hover.chunkIdx} · {hover.timeStart.toFixed(1)}–{hover.timeEnd.toFixed(1)}с
            </span>
            <span className="ml-auto text-yellow-300 font-semibold">
              {(hover.prob * 100).toFixed(1)}%
            </span>
          </div>
        ) : (
          <p className="text-white/30 text-xs italic">
            Наведіть на сегмент, щоб побачити час, клас і ймовірність
          </p>
        )}
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
                className="flex-1 flex flex-col-reverse h-full relative"
              >
                {segments.map(({ cls, p }) => (
                  <div
                    key={cls}
                    style={{ height: `${p * 100}%`, background: colorFor(cls) }}
                    className="relative transition-all hover:brightness-125 cursor-pointer"
                    onMouseEnter={() =>
                      setHover({
                        chunkIdx: c.chunk_idx,
                        timeStart: c.time_start,
                        timeEnd: c.time_end,
                        cls,
                        prob: p,
                      })
                    }
                    onMouseLeave={() => setHover(null)}
                  />
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
