import { PATTERNS, PATTERN_ORDER } from "../data/patterns";
import PatternDisplay, { PatternLegend } from "./PatternDisplay";

export default function PatternsGallery() {
  const strumPatterns = PATTERN_ORDER
    .map((k) => PATTERNS[k])
    .filter((p) => p && p.kind === "strum");
  const arpPatterns = PATTERN_ORDER
    .map((k) => PATTERNS[k])
    .filter((p) => p && p.kind === "arp");

  return (
    <div className="space-y-6">
      {/* Strumming section */}
      <section>
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <span className="w-1 h-5 bg-yellow-400 rounded-full" />
            Бої
          </h2>
          <PatternLegend kind="strum" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {strumPatterns.map((p) => (
            <div key={p.label} className="card">
              <div className="flex items-baseline justify-between mb-3">
                <h3 className="font-semibold text-white">{p.name}</h3>
                <span className="text-xs text-yellow-300/80 font-mono">{p.label}</span>
              </div>
              <div className="py-2">
                <PatternDisplay pattern={p} size="sm" />
              </div>
              <p className="mt-3 text-xs text-white/50 font-mono">{p.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Arpeggio section */}
      <section>
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <span className="w-1 h-5 bg-emerald-500 rounded-full" />
            Арпеджіо
          </h2>
          <PatternLegend kind="arp" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {arpPatterns.map((p) => (
            <div key={p.label} className="card">
              <div className="flex items-baseline justify-between mb-3">
                <h3 className="font-semibold text-white">{p.name}</h3>
                <span className="text-xs text-emerald-300/80 font-mono">{p.label}</span>
              </div>
              <div className="py-2">
                <PatternDisplay pattern={p} size="sm" />
              </div>
              <p className="mt-3 text-xs text-white/50 font-mono">{p.description}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
