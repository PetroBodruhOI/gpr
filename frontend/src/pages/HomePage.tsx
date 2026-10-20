import type { ReactNode } from "react";
import UploadForm from "../components/UploadForm";
import PipelineStage from "../components/PipelineStage";
import PatternDisplay from "../components/PatternDisplay";
import { ArrowRightIcon } from "../components/icons";
import { colorFor, PATTERNS, PATTERN_ORDER } from "../data/patterns";
import { href, useReveal } from "../useHashRoute";

function Reveal({ children, className = "" }: { children: ReactNode; className?: string }) {
  const ref = useReveal<HTMLElement>();
  return <section ref={ref} className={`reveal group ${className}`}>{children}</section>;
}

// ── How it works: each stage has a small visual that plays once on reveal ──

const STEMS = [
  { name: "Вокал", keep: false },
  { name: "Барабани", keep: false },
  { name: "Бас", keep: false },
  { name: "Гітара", keep: true },
];

function StemsVisual() {
  return (
    <div className="space-y-2.5">
      {STEMS.map((s, i) => (
        <div
          key={s.name}
          className={`flex items-center gap-3 transition-opacity duration-700 ${s.keep ? "" : "group-data-[inview]:opacity-25"}`}
          style={{ transitionDelay: `${400 + i * 150}ms` }}
        >
          <span className="w-20 text-sm text-white/70">{s.name}</span>
          <span className={`h-3 flex-1 rounded-full ${s.keep ? "bg-accent" : "bg-white/30"}`} />
        </div>
      ))}
    </div>
  );
}

function BeatsVisual() {
  return (
    <div>
      <div className="flex items-end gap-2 h-20">
        {Array.from({ length: 16 }, (_, i) => (
          <span
            key={i}
            className={`flex-1 rounded-full origin-bottom scale-y-0 transition-transform duration-500 ease-out group-data-[inview]:scale-y-100 ${
              i % 4 === 0 ? "h-full bg-accent" : "h-1/2 bg-glow"
            }`}
            style={{ transitionDelay: `${300 + i * 60}ms` }}
          />
        ))}
      </div>
      <div className="mt-3 flex justify-between text-xs text-white/55">
        <span>такт 1</span><span>такт 2</span><span>такт 3</span><span>такт 4</span>
      </div>
    </div>
  );
}

// The model weighs every pattern in the library and picks one: a highlight
// runs across all six (same neutral colour), then settles on the winner.
const PICKED = "6b";
const SCAN_STEP_MS = 140;

function PickVisual() {
  const settleMs = 300 + PATTERN_ORDER.length * SCAN_STEP_MS + 150;
  return (
    <div>
      <div className="grid grid-cols-3 gap-2">
        {PATTERN_ORDER.map((code, i) => (
          <span
            key={code}
            className="relative overflow-hidden py-2 rounded-lg text-center font-display font-bold border border-white/10 bg-raised text-white/70 group-data-[inview]:animate-[scan_420ms_ease-out_both]"
            style={{ animationDelay: `${300 + i * SCAN_STEP_MS}ms` }}
          >
            {code === PICKED && (
              <span
                className="absolute inset-0 bg-accent opacity-0 group-data-[inview]:animate-[pick-fill_300ms_ease-out_both]"
                style={{ animationDelay: `${settleMs}ms` }}
              />
            )}
            <span
              className={`relative ${code === PICKED ? "group-data-[inview]:animate-[pick-text_300ms_ease-out_both]" : ""}`}
              style={code === PICKED ? { animationDelay: `${settleMs}ms` } : undefined}
            >
              {code}
            </span>
          </span>
        ))}
      </div>
      <p
        className="mt-4 text-sm text-white/70 opacity-0 group-data-[inview]:animate-[pick-fill_300ms_ease-out_both]"
        style={{ animationDelay: `${settleMs}ms` }}
      >
        Обрано: <span className="font-semibold text-white">{PATTERNS[PICKED].name}</span>
      </p>
    </div>
  );
}

const STAGES = [
  {
    title: "Виділяє гітару",
    text: "Нейромережа HTDemucs розкладає запис на окремі доріжки — вокал, барабани, бас, гітару — і залишає тільки гітару.",
    visual: <StemsVisual />,
  },
  {
    title: "Знаходить ритм",
    text: "Модель BeatThis позначає кожну долю й початок такту, тож GPR знає, де «раз», а де «і».",
    visual: <BeatsVisual />,
  },
  {
    title: "Підбирає патерн",
    text: "Модель оцінює кожен із шести патернів бібліотеки й обирає той, що найкраще підходить до вашого треку.",
    visual: <PickVisual />,
  },
];

export default function HomePage({ onStart }: { onStart: (taskId: string) => void }) {
  return (
    <div className="space-y-24 sm:space-y-32">
      {/* ── Hero ── */}
      <section className="grid lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] gap-10 lg:gap-14 items-center pt-4 sm:pt-8">
        <div className="enter">
          <h1 className="text-5xl sm:text-6xl xl:text-7xl font-extrabold leading-[1.02] tracking-tight text-balance">
            Почуй пісню&nbsp;— <span className="block text-accent">отримай бій</span>
          </h1>
          <p className="mt-6 text-lg sm:text-xl text-white/75 leading-relaxed max-w-xl text-pretty">
            Завантажте запис улюбленої пісні. GPR виділить гітару, знайде ритм і покаже,
            як її грати: стрілками, по долях.
          </p>
          <div className="mt-8 card p-4 sm:p-5">
            <UploadForm onStart={onStart} />
          </div>
          <p className="mt-3 text-sm text-white/50">
            Обробка триває від пів хвилини до кількох хвилин — залежно від довжини треку.
          </p>
        </div>
        <PipelineStage className="enter [--i:2]" />
      </section>

      {/* ── How it works ── */}
      <div>
        <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight">Як GPR чує ваш трек</h2>
        <Reveal className="mt-10 rounded-3xl bg-surface border border-white/[0.06] grid lg:grid-cols-3">
          {STAGES.map((s, i) => (
            <div
              key={s.title}
              className="relative flex flex-col gap-6 p-6 sm:p-8 border-white/[0.08] [&:not(:first-child)]:border-t lg:[&:not(:first-child)]:border-t-0 lg:[&:not(:first-child)]:border-l"
            >
              {i > 0 && (
                <span className="absolute z-10 w-8 h-8 rounded-full bg-accent text-ink flex items-center justify-center left-1/2 -top-4 -translate-x-1/2 rotate-90 lg:rotate-0 lg:left-0 lg:top-1/2 lg:-translate-y-1/2">
                  <ArrowRightIcon />
                </span>
              )}
              <div>
                <h3 className="text-2xl font-bold">
                  <span className="text-accent">{i + 1}.</span> {s.title}
                </h3>
                <p className="mt-3 text-white/70 leading-relaxed">{s.text}</p>
              </div>
              <div className="mt-auto pt-2">{s.visual}</div>
            </div>
          ))}
        </Reveal>
      </div>

      {/* ── Why ── */}
      <Reveal className="grid lg:grid-cols-2 gap-10 items-center">
        <div>
          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-balance">
            Не існує єдиного правильного способу зіграти пісню
          </h2>
          <p className="mt-5 text-lg text-white/75 leading-relaxed">
            GPR порекомендує патерн, який найкраще підходить до вашого треку, — або кілька
            найдоречніших варіантів, якщо модель не впевнена.
          </p>
          <ul className="mt-6 space-y-3 text-white/85">
            {[
              "Рекомендований патерн і наскільки модель у ньому впевнена",
              "Як грати: кожен удар по долях — вниз, вгору чи пауза",
              "Як патерн змінюється протягом пісні, сегмент за сегментом",
            ].map((t) => (
              <li key={t} className="flex gap-3">
                <span className="mt-2 w-2 h-2 rounded-full bg-accent flex-shrink-0" />
                {t}
              </li>
            ))}
          </ul>
        </div>
        <div className="card-featured">
          <div className="flex items-baseline justify-between gap-4">
            <h3 className="text-3xl font-extrabold">Розгорнутий бій</h3>
            <span className="font-display text-2xl font-bold text-accent tabular-nums">74%</span>
          </div>
          <div className="mt-5 overflow-x-auto"><PatternDisplay pattern={PATTERNS["8b"]} size="md" /></div>
          <div className="mt-6 flex h-3 rounded-full overflow-hidden gap-0.5">
            {["8b", "8b", "6b", "8b", "8b", "6x"].map((c, i) => (
              <span key={i} className="flex-1" style={{ background: colorFor(c) }} />
            ))}
          </div>
          <div className="mt-2 flex justify-between text-xs text-white/55 tabular-nums">
            <span>0:00</span><span>по ходу треку</span><span>0:27</span>
          </div>
          <p className="mt-5 pt-4 border-t border-white/[0.08] text-sm text-white/55">
            Приклад: так виглядає відповідь GPR.
          </p>
        </div>
      </Reveal>

      {/* ── Patterns ── */}
      <Reveal>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight">Шість патернів, які знає GPR</h2>
          <a href={href("patterns")} className="btn-secondary">
            Усі патерни <ArrowRightIcon />
          </a>
        </div>
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {PATTERN_ORDER.map((k) => {
            const p = PATTERNS[k];
            return (
              <a key={k} href={href("patterns")} className="card block transition-colors duration-150 hover:border-white/20">
                <div className="flex items-baseline justify-between gap-3">
                  <h3 className="font-bold text-lg">{p.name}</h3>
                  <span className="font-display font-bold text-sm" style={{ color: p.color }}>{p.label}</span>
                </div>
                <div className="mt-4 overflow-x-auto"><PatternDisplay pattern={p} size="sm" /></div>
              </a>
            );
          })}
        </div>
      </Reveal>

      {/* ── Closing call to action ── */}
      <Reveal className="rounded-3xl bg-accent text-ink px-6 py-12 sm:px-12 sm:py-16 grid gap-6 lg:grid-cols-[1fr_auto] items-center">
        <div>
          <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-balance">Спробуйте зі своєю піснею</h2>
          <p className="mt-3 text-lg text-ink/80 max-w-xl">
            Аудіо або відео з телефону — MP3, WAV, M4A, MOV. Без реєстрації.
          </p>
        </div>
        <a href={href("detect")} className="btn px-7 py-4 text-lg bg-ink text-white hover:bg-surface">
          Визначити патерн <ArrowRightIcon className="w-5 h-5" />
        </a>
      </Reveal>
    </div>
  );
}
