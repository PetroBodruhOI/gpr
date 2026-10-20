import { Pattern, PATTERNS, PATTERN_ORDER } from "../data/patterns";
import PatternDisplay, { PatternLegend } from "./PatternDisplay";

function PatternTile({ pattern }: { pattern: Pattern }) {
  return (
    <div className="card flex flex-col">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="font-semibold text-white">{pattern.name}</h3>
        <span className="text-xs text-white/45 font-mono">{pattern.label}</span>
      </div>
      <div className="my-5 py-1 overflow-x-auto">
        <PatternDisplay pattern={pattern} size="md" />
      </div>
      <p className="mt-auto text-xs text-white/45 font-mono tracking-wide">{pattern.description}</p>
    </div>
  );
}

function Section({ title, kind }: { title: string; kind: Pattern["kind"] }) {
  const patterns = PATTERN_ORDER.map((k) => PATTERNS[k]).filter((p) => p && p.kind === kind);
  return (
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-3 mb-4">
        <h2 className="text-xl font-semibold text-white">{title}</h2>
        <PatternLegend kind={kind} />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {patterns.map((p) => <PatternTile key={p.label} pattern={p} />)}
      </div>
    </section>
  );
}

export default function PatternsGallery() {
  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">Список патернів</h1>
        <p className="mt-2 text-white/60 max-w-2xl">
          Шість патернів, які розпізнає модель: три бої та три арпеджіо.
        </p>
      </div>
      <Section title="Бої" kind="strum" />
      <Section title="Арпеджіо" kind="arp" />
    </div>
  );
}
