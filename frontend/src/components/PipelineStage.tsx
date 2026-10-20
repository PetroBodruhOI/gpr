import { useState, type CSSProperties } from "react";
import { PATTERNS, StrumStep } from "../data/patterns";
import { ChevronDownIcon } from "./icons";

// The home-page explainer, read top to bottom:
//   1 the song's waveform → 2 the same waveform with only the guitar hits lit
//   → each hit drops a line into 3 its beat cell, which strikes the arrow.
// One 4 s clock drives everything (see .stage-* in index.css): a playhead
// sweeps all three lanes, and a hit, its line and its cell light up the moment
// the playhead reaches that column. Each loop shows the next strum pattern.

const LOOP_S = 4;
const BARS = 48;
const DEMOS = ["6b", "8b", "6x"];
const COUNT = ["1", "&", "2", "&", "3", "&", "4", "&"];

// Deterministic "random" heights so the waveform looks like music.
const heights = Array.from({ length: BARS }, (_, i) => 28 + Math.round(60 * Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.6))));

const at = (column: number): CSSProperties => ({ animationDelay: `${(((column + 0.5) / 8) * LOOP_S).toFixed(2)}s` });

const ARROW: Record<Exclude<StrumStep, "-">, { glyph: string; cls: string }> = {
  D: { glyph: "↓", cls: "text-accent" },
  U: { glyph: "↑", cls: "text-glow" },
  u: { glyph: "↑", cls: "text-glow" },
  X: { glyph: "×", cls: "text-red-300" },
};

function LaneLabel({ children }: { children: React.ReactNode }) {
  return <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-glow truncate">{children}</div>;
}

function Waveform({ dim = false, className = "" }: { dim?: boolean; className?: string }) {
  return (
    <div className={`flex items-center gap-[3px] ${className}`}>
      {heights.map((h, i) => (
        <span
          key={i}
          className={`flex-1 rounded-full ${dim ? "bg-white/[0.12]" : "stage-bar bg-white/70"}`}
          style={{ height: `${h}%`, animationDelay: dim ? undefined : `${-(i % 7) * 0.13}s` }}
        />
      ))}
    </div>
  );
}

export default function PipelineStage({ className = "" }: { className?: string }) {
  const [demo, setDemo] = useState(0);
  const pattern = PATTERNS[DEMOS[demo]];
  const steps: StrumStep[] = pattern.kind === "strum" ? pattern.steps : [];

  return (
    <div
      role="img"
      aria-label="Анімація: з хвилі пісні виділяється гітара, і кожен удар гітари стає стрілкою бою на своїй долі."
      className={`relative rounded-3xl bg-surface border border-white/[0.06] p-5 sm:p-7 overflow-hidden ${className}`}
    >
      {/* Firelight behind the guitar lane */}
      <div className="pointer-events-none absolute inset-x-0 top-[30%] h-1/2 bg-[radial-gradient(50%_60%_at_50%_50%,rgba(255,122,26,0.16),transparent)]" />

      <div className="relative">
        {/* 1 · the song */}
        <LaneLabel>1 · Пісня</LaneLabel>
        <Waveform className="h-16" />

        <div className="my-3 flex items-center gap-1.5 text-xs text-white/60">
          <ChevronDownIcon className="w-4 h-4 text-glow" />
          HTDemucs прибирає вокал, барабани й бас
        </div>

        {/* 2 · the same song, only the guitar left — and 3 · its beats */}
        <LaneLabel>2 · Тільки гітара</LaneLabel>
        <div key={demo} className="relative">
          <Waveform dim className="absolute inset-x-0 top-0 h-14" />
          <div className="relative grid grid-cols-8">
            {steps.map((step, i) => (
              <div key={i} className="flex flex-col items-center px-[3px]">
                <div className="h-14 flex items-center">
                  {step !== "-" && <span className="stage-transient block w-1.5 h-full rounded-full" style={at(i)} />}
                </div>
                {step !== "-" ? (
                  <span className="stage-transient block w-0.5 h-7" style={at(i)} />
                ) : (
                  <span className="block h-7" />
                )}
                <span className="my-1 text-xs text-white/50 font-display">{COUNT[i]}</span>
                <div className="stage-cell w-full aspect-[3/4] rounded-xl bg-raised flex items-center justify-center" style={at(i)}>
                  {step === "-" ? (
                    <span className="text-white/25 text-xl">·</span>
                  ) : (
                    <span className={`stage-arrow font-display font-bold text-3xl sm:text-4xl ${ARROW[step].cls}`} style={at(i)}>
                      {ARROW[step].glyph}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-4 text-xs text-white/60 truncate">
          <span className="font-semibold uppercase tracking-[0.14em] text-glow">3 · Як грати</span>
          <span className="mx-1.5">—</span>
          приклад: <span className="text-white/85">{pattern.name}</span>
        </div>

        {/* Playhead across all three lanes; each pass shows the next pattern */}
        <div
          className="stage-playhead pointer-events-none absolute inset-x-0 top-0 bottom-8"
          onAnimationIteration={() => setDemo((d) => (d + 1) % DEMOS.length)}
        >
          <div className="absolute inset-y-0 left-0 w-0.5 bg-glow/80 rounded-full" />
        </div>
      </div>
    </div>
  );
}
