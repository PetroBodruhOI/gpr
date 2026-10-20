import { CheckIcon } from "./icons";

interface Props {
  progress: number;
  message: string;
}

// Backend progress bands (backend/app.py run_predict):
// 5-50 Demucs → 55-69 BeatThis → 70-95 features/classification → 100 done.
const STAGES = [
  { from: 0, label: "Виділення гітари" },
  { from: 55, label: "Аналіз ритму" },
  { from: 70, label: "Підбір патерну" },
];

export default function ProgressBar({ progress, message }: Props) {
  const current = STAGES.reduce((idx, s, i) => (progress >= s.from ? i : idx), 0);

  return (
    <div className="card">
      <div className="flex items-baseline justify-between gap-4 mb-4">
        <h3 className="font-semibold text-white">Обробка</h3>
        <span className="text-2xl font-semibold text-accent tabular-nums">{progress}%</span>
      </div>

      <div
        role="progressbar"
        aria-label="Прогрес обробки"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progress}
        className="h-1.5 rounded-full bg-raised overflow-hidden"
      >
        <div
          className="h-full w-full bg-glow origin-left transition-transform duration-500 ease-out"
          style={{ transform: `scaleX(${progress / 100})` }}
        />
      </div>

      <ol className="mt-5 grid grid-cols-3 gap-2 text-xs sm:text-sm">
        {STAGES.map((s, i) => {
          const state = i < current ? "done" : i === current ? "active" : "todo";
          return (
            <li key={s.label} className="flex items-center gap-2 min-w-0">
              <span
                className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center border ${
                  state === "done"
                    ? "bg-glow/20 border-glow/60 text-glow"
                    : state === "active"
                    ? "border-accent/70"
                    : "border-white/15"
                }`}
              >
                {state === "done" && <CheckIcon className="w-3 h-3" />}
                {state === "active" && <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />}
              </span>
              <span className={`truncate ${state === "todo" ? "text-white/35" : "text-white/80"}`}>
                {s.label}
              </span>
            </li>
          );
        })}
      </ol>

      <p aria-live="polite" className="mt-4 text-white/55 text-sm min-h-[1.25rem]">{message}</p>
    </div>
  );
}
