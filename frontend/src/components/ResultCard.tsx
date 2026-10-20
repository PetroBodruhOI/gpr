import { useState } from "react";
import { PredictResult, submitFeedback } from "../api/client";
import { colorFor, PATTERNS } from "../data/patterns";
import PatternDisplay, { PatternLegend } from "./PatternDisplay";
import ChunkDistributionChart from "./ChunkDistributionChart";
import {
  ArrowLeftIcon, CheckIcon, ChevronDownIcon, ChevronUpIcon, ThumbDownIcon, ThumbUpIcon,
} from "./icons";

interface Props {
  result: PredictResult;
  taskId: string;
  onReset?: () => void;
}

function FeedbackPanel({ taskId }: { taskId: string }) {
  const [submitted, setSubmitted] = useState<"good" | "bad" | null>(null);
  const [busy, setBusy] = useState(false);

  const send = async (rating: "good" | "bad") => {
    if (busy || submitted) return;
    setBusy(true);
    try {
      await submitFeedback(taskId, rating);
      setSubmitted(rating);
    } catch (e) {
      console.error("feedback failed", e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card text-center" aria-live="polite">
      {submitted ? (
        <>
          <p className="flex items-center justify-center gap-2 text-glow font-semibold">
            <CheckIcon className="w-5 h-5" />
            Дякуємо за відгук!
          </p>
          <p className="mt-1 text-white/50 text-sm">Ваша оцінка допомагає нам покращити модель.</p>
        </>
      ) : (
        <>
          <h3 className="section-title">Допоможіть нам покращити модель</h3>
          <p className="mt-1 mb-4 text-white/60 text-sm">
            Чи вважаєте, що рекомендація була правильною?
          </p>
          <div className="flex gap-3 justify-center flex-wrap">
            <button
              onClick={() => send("good")}
              disabled={busy}
              className="btn px-5 py-2.5 text-glow bg-glow/10 border border-glow/40
                         hover:bg-glow/20 disabled:opacity-50"
            >
              <ThumbUpIcon />
              Так, підходить
            </button>
            <button
              onClick={() => send("bad")}
              disabled={busy}
              className="btn px-5 py-2.5 text-red-200 bg-red-500/10 border border-red-500/40
                         hover:bg-red-500/20 disabled:opacity-50"
            >
              <ThumbDownIcon />
              Ні, не те
            </button>
          </div>
        </>
      )}
    </div>
  );
}

const TIMELINE_PREVIEW_COUNT = 2;

function TimelineTable({ chunks, total }: { chunks: PredictResult["chunks"]; total: number }) {
  const [expanded, setExpanded] = useState(false);
  const canCollapse = chunks.length > TIMELINE_PREVIEW_COUNT;
  const visible = expanded || !canCollapse ? chunks : chunks.slice(0, TIMELINE_PREVIEW_COUNT);
  const hiddenCount = chunks.length - TIMELINE_PREVIEW_COUNT;

  const th = "px-3 sm:px-4 py-2.5 text-left font-medium text-white/50 text-xs";
  const td = "px-3 sm:px-4 py-3";

  return (
    <div className="card">
      <h3 className="section-title mb-4">
        Часова шкала
        <span className="ml-2 text-white/40 text-sm font-normal">{total} сегментів</span>
      </h3>
      <div className="overflow-x-auto rounded-lg border border-white/[0.06]">
        <table className="w-full text-sm tabular-nums">
          <thead>
            <tr className="bg-ink/50 border-b border-white/[0.06]">
              <th className={`${th} hidden sm:table-cell w-12`}>#</th>
              <th className={th}>Час</th>
              <th className={th}>Патерн</th>
              <th className={`${th} text-right`}>
                <span className="hidden sm:inline">Впевненість</span>
                <span className="sm:hidden" aria-label="Впевненість">%</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.map((c) => (
              <tr key={c.chunk_idx} className="border-b border-white/[0.05] last:border-0">
                <td className={`${td} hidden sm:table-cell text-white/40`}>{c.chunk_idx}</td>
                <td className={`${td} text-white/70 whitespace-nowrap`}>
                  {c.time_start.toFixed(1)}–{c.time_end.toFixed(1)} с
                </td>
                <td className={td}>
                  <span className="flex items-center gap-2 min-w-0">
                    <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: colorFor(c.label) }} />
                    <span className="text-white font-medium truncate">{PATTERNS[c.label]?.name ?? c.label}</span>
                    <span className="hidden sm:inline text-white/40 font-mono text-xs">{c.label}</span>
                  </span>
                </td>
                <td className={`${td} text-right text-accent font-semibold`}>
                  {(c.confidence * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {canCollapse && (
        <div className="mt-4 flex justify-center">
          <button onClick={() => setExpanded((v) => !v)} className="btn-secondary text-sm">
            {expanded ? "Показати менше" : `Показати більше (${hiddenCount})`}
            {expanded ? <ChevronUpIcon /> : <ChevronDownIcon />}
          </button>
        </div>
      )}
    </div>
  );
}

const HIGH_CONFIDENCE_THRESHOLD = 0.75;     // 75%
const MIN_SHOW_THRESHOLD = 0.25;             // 25%
const MAX_ALT_RECOMMENDATIONS = 3;

export function pickRecommendations(
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

  // Low confidence → all classes with at least 25% confidence,
  // capped at MAX_ALT_RECOMMENDATIONS. Always show at least the top class.
  const candidates = sorted
    .filter((s) => s.prob >= MIN_SHOW_THRESHOLD)
    .slice(0, MAX_ALT_RECOMMENDATIONS);
  return candidates.length > 0 ? candidates : [sorted[0]];
}

function PatternCard({ label, prob, primary }: { label: string; prob: number; primary: boolean }) {
  const pattern = PATTERNS[label];

  return (
    <div className={primary ? "card-featured" : "card"}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className={`font-bold text-white tracking-tight ${primary ? "text-2xl sm:text-3xl" : "text-xl"}`}>
          {pattern ? pattern.name : label}
        </h3>
        {pattern && <span className="text-sm text-white/45 font-mono">{pattern.label}</span>}
        <span className="ml-auto flex items-baseline gap-1.5">
          <span className={`font-semibold text-accent tabular-nums ${primary ? "text-2xl" : "text-xl"}`}>
            {(prob * 100).toFixed(1)}%
          </span>
          <span className="text-white/50 text-sm">впевненість</span>
        </span>
      </div>

      {pattern ? (
        <>
          <p className="mt-1 text-white/45 text-xs font-mono tracking-wide">{pattern.description}</p>
          <div className="mt-5 py-1 overflow-x-auto">
            <PatternDisplay pattern={pattern} size={primary ? "lg" : "md"} />
          </div>
          <div className="mt-4 pt-4 border-t border-white/[0.06]">
            <PatternLegend kind={pattern.kind} />
          </div>
        </>
      ) : (
        <p className="mt-3 text-white/45 text-sm">Опис патерну для «{label}» не знайдено</p>
      )}
    </div>
  );
}

export default function ResultCard({ result, taskId, onReset }: Props) {
  const recommendations = pickRecommendations(result.mean_probs);
  const isHighConf = result.final_conf >= HIGH_CONFIDENCE_THRESHOLD;

  // Each block enters once, staggered: header, cards, chart, table, feedback.
  let i = 0;
  const enter = () => ({ className: "enter", style: { "--i": i++ } as React.CSSProperties });

  return (
    <div className="space-y-5">
      <div {...enter()} className="enter flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <span className="w-10 h-10 flex-shrink-0 rounded-lg bg-glow/15 border border-glow/30 text-glow flex items-center justify-center">
            <CheckIcon className="w-5 h-5" />
          </span>
          <div className="min-w-0">
            <h2 className={`text-xl font-semibold ${isHighConf ? "text-white" : "text-accent"}`}>
              {isHighConf ? "Рекомендований патерн" : "Можливі патерни"}
            </h2>
            <p className="text-white/55 text-sm">
              {isHighConf
                ? "Найкращий збіг для вашого треку"
                : recommendations.length > 1
                ? `Модель не впевнена — показуємо топ-${recommendations.length} варіанти`
                : "Модель не впевнена — це найімовірніший варіант"}
            </p>
          </div>
        </div>
        {onReset && (
          <button onClick={onReset} className="btn-secondary hidden sm:inline-flex text-sm flex-shrink-0">
            <ArrowLeftIcon />
            Інший трек
          </button>
        )}
      </div>

      {recommendations.map((r, idx) => (
        <div key={r.label} {...enter()}>
          <PatternCard label={r.label} prob={r.prob} primary={idx === 0} />
        </div>
      ))}

      <div {...enter()}><ChunkDistributionChart chunks={result.chunks} /></div>
      <div {...enter()}><TimelineTable chunks={result.chunks} total={result.n_chunks} /></div>
      <div {...enter()}><FeedbackPanel taskId={taskId} /></div>
    </div>
  );
}
