import { Pattern, StrumStep, ArpStep } from "../data/patterns";

interface Props {
  pattern: Pattern;
  size?: "sm" | "md" | "lg";
}

const SIZES = {
  sm: { cell: "w-7 h-10", icon: "text-base", label: "text-[10px]" },
  md: { cell: "w-9 h-14", icon: "text-xl",  label: "text-xs"     },
  lg: { cell: "w-12 h-16", icon: "text-2xl", label: "text-sm"     },
};

function StrumIcon({ kind, sizeClass }: { kind: StrumStep; sizeClass: string }) {
  const base = `${sizeClass} font-bold`;
  if (kind === "D") return <span className={`${base} text-yellow-300`}>↓</span>;
  if (kind === "U") return <span className={`${base} text-emerald-400`}>↑</span>;
  if (kind === "u") return <span className={`${base} text-sm text-emerald-400`}>↑</span>;
  if (kind === "X") return <span className={`${base} text-red-400`}>×</span>;
  return <span className={`${base} text-white/20`}>·</span>;
}

function StrumGrid({ steps, size }: { steps: StrumStep[]; size: "sm" | "md" | "lg" }) {
  const s = SIZES[size];
  // Standard 4/4 counting: "1 & 2 & 3 & 4 &" — show up to 8 labels
  const beatLabels = ["1", "&", "2", "&", "3", "&", "4", "&"];

  return (
    <div className="flex gap-1 justify-center">
      {steps.map((step, i) => (
        <div key={i} className="flex flex-col items-center">
          <div className={`${s.label} text-white/40 mb-1 font-mono`}>
            {beatLabels[i] || ""}
          </div>
          <div className={`${s.cell} flex items-center justify-center rounded-md
                          bg-slate-900/40 border border-white/10`}>
            <StrumIcon kind={step} sizeClass={s.icon} />
          </div>
        </div>
      ))}
    </div>
  );
}

function ArpStepBox({ step, size }: { step: ArpStep; size: "sm" | "md" | "lg" }) {
  const s = SIZES[size];
  const isChord = step.strings.length > 1;
  const isEmpty = step.strings.length === 0;

  if (isEmpty) {
    return (
      <div className={`${s.cell} flex items-center justify-center rounded-md
                       bg-slate-900/40 border border-white/10`}>
        <span className="text-white/20">·</span>
      </div>
    );
  }

  if (step.isBass) {
    return (
      <div className={`${s.cell} flex flex-col items-center justify-center rounded-md
                       bg-yellow-400/10 border border-yellow-400/40`}>
        <span className={`${s.label} text-yellow-300/70 leading-none`}></span>
        <span className={`${s.icon} font-bold text-yellow-300 leading-none`}>B</span>
      </div>
    );
  }

  if (isChord) {
    return (
      <div className={`${s.cell} flex flex-col items-center justify-center rounded-md
                       bg-emerald-500/10 border border-emerald-500/40 gap-0.5`}>
        {step.strings.map((s_) => (
          <span key={s_} className={`${SIZES[size].label} font-bold text-emerald-300 leading-none`}>
            {s_}
          </span>
        ))}
      </div>
    );
  }

  // single string
  return (
    <div className={`${s.cell} flex items-center justify-center rounded-md
                     bg-emerald-500/10 border border-emerald-500/30`}>
      <span className={`${s.icon} font-bold text-emerald-300`}>{step.strings[0]}</span>
    </div>
  );
}

function ArpGrid({ steps, size }: { steps: ArpStep[]; size: "sm" | "md" | "lg" }) {
  return (
    <div className="flex gap-1 justify-center flex-wrap">
      {steps.map((step, i) => (
        <ArpStepBox key={i} step={step} size={size} />
      ))}
    </div>
  );
}

export default function PatternDisplay({ pattern, size = "md" }: Props) {
  return (
    <div className="flex flex-col gap-2">
      {pattern.kind === "strum" ? (
        <StrumGrid steps={pattern.steps} size={size} />
      ) : (
        <ArpGrid steps={pattern.steps} size={size} />
      )}
    </div>
  );
}

// Re-export legend for use in other components
export function PatternLegend({ kind }: { kind: "strum" | "arp" }) {
  if (kind === "strum") {
    return (
      <div className="flex flex-wrap gap-3 text-xs text-white/50">
        <span><span className="text-yellow-300 font-bold">↓</span> вниз</span>
        <span><span className="text-emerald-400 font-bold">↑</span> вгору</span>
        <span><span className="text-red-400 font-bold">×</span> приглушення</span>
        <span><span className="text-white/30">·</span> пауза</span>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap gap-3 text-xs text-white/50">
      <span><span className="text-yellow-300 font-bold">B</span> басова нота</span>
      <span><span className="text-emerald-300 font-bold">1–6</span> номер струни</span>
      <span>стовпчик = акорд</span>
    </div>
  );
}
