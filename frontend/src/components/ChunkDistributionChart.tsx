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

const GRIDLINES = [100, 75, 50, 25, 0];

export default function ChunkDistributionChart({ chunks }: Props) {
  const [hover, setHover] = useState<HoverInfo | null>(null);

  // Every class that appears anywhere; known patterns first, then alphabetical.
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
      <h3 className="section-title">Розподіл за часом</h3>
      <p className="mt-1 mb-4 text-white/50 text-sm">
        Ймовірності кожного класу в усіх часових сегментах
      </p>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-2 mb-4">
        {allClasses.map((cls) => (
          <div key={cls} className="flex items-center gap-1.5 text-xs">
            <span className="w-3 h-3 rounded-sm" style={{ background: colorFor(cls) }} />
            <span className="font-mono text-white/80">{cls}</span>
            {PATTERNS[cls] && (
              <span className="text-white/40 hidden sm:inline">{PATTERNS[cls].name}</span>
            )}
          </div>
        ))}
      </div>

      {/* Readout — fixed height so the layout doesn't jump */}
      <div className="h-11 mb-3 flex items-center" aria-live="polite">
        {hover ? (
          <div className="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg bg-ink/70 border border-white/10 text-sm">
            <span className="w-3 h-3 rounded-sm flex-shrink-0" style={{ background: colorFor(hover.cls) }} />
            <span className="font-mono text-white/90">{hover.cls}</span>
            {PATTERNS[hover.cls] && (
              <span className="text-white/60 text-xs truncate">{PATTERNS[hover.cls].name}</span>
            )}
            <span className="text-white/40 text-xs ml-1 hidden sm:inline tabular-nums">
              сегмент #{hover.chunkIdx} · {hover.timeStart.toFixed(1)}–{hover.timeEnd.toFixed(1)} с
            </span>
            <span className="ml-auto text-accent font-semibold tabular-nums">
              {(hover.prob * 100).toFixed(1)}%
            </span>
          </div>
        ) : (
          <p className="text-white/35 text-xs">
            Наведіть на сегмент або торкніться його, щоб побачити час, клас і ймовірність
          </p>
        )}
      </div>

      {/* Y axis + stacked columns */}
      <div className="flex gap-2">
        <div className="w-9 h-56 flex flex-col justify-between text-[10px] text-white/35 tabular-nums text-right -my-1.5 py-0">
          {GRIDLINES.map((v) => <span key={v}>{v}%</span>)}
        </div>
        <div className="relative flex-1 h-56">
          <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
            {GRIDLINES.map((v) => <div key={v} className="border-t border-white/[0.06]" />)}
          </div>
          <div className="relative flex h-full rounded-md overflow-hidden border border-white/[0.06]">
            {chunks.map((c) => {
              const segments = allClasses
                .map((cls) => ({ cls, p: c.probs[cls] ?? 0 }))
                .filter((s) => s.p > 0);
              return (
                <div key={c.chunk_idx} className="flex-1 flex flex-col-reverse h-full">
                  {segments.map(({ cls, p }) => {
                    const info = {
                      chunkIdx: c.chunk_idx, timeStart: c.time_start, timeEnd: c.time_end, cls, prob: p,
                    };
                    return (
                      <div
                        key={cls}
                        style={{ height: `${p * 100}%`, background: colorFor(cls) }}
                        className={`cursor-pointer transition-[filter] duration-150 ${
                          hover && hover.chunkIdx === c.chunk_idx && hover.cls === cls ? "brightness-125" : ""
                        }`}
                        onMouseEnter={() => setHover(info)}
                        onMouseLeave={() => setHover(null)}
                        onClick={() => setHover(info)}
                      />
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
