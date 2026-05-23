import { useState } from "react";
import { predictUrl } from "../api/client";

interface Props {
  onStart: (taskId: string) => void;
}

export default function UrlForm({ onStart }: Props) {
  const [url, setUrl] = useState("");
  const [startSec, setStartSec] = useState<string>("");
  const [durSec, setDurSec] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!url) return;
    setBusy(true);
    try {
      const taskId = await predictUrl(
        url,
        startSec ? Number(startSec) : undefined,
        durSec ? Number(durSec) : undefined,
      );
      onStart(taskId);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-5">
      <div>
        <label className="block text-xs font-semibold text-white/50 mb-2 uppercase tracking-wider">
          YouTube URL
        </label>
        <input
          type="url"
          placeholder="https://youtu.be/dQw4w9WgXcQ"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={busy}
          className="input-field"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-semibold text-white/50 mb-2 uppercase tracking-wider">
            Start (sec)
          </label>
          <input
            type="number"
            placeholder="0"
            value={startSec}
            onChange={(e) => setStartSec(e.target.value)}
            disabled={busy}
            className="input-field"
          />
        </div>
        <div>
          <label className="block text-xs font-semibold text-white/50 mb-2 uppercase tracking-wider">
            Duration (sec)
          </label>
          <input
            type="number"
            placeholder="30"
            value={durSec}
            onChange={(e) => setDurSec(e.target.value)}
            disabled={busy}
            className="input-field"
          />
        </div>
      </div>

      <button
        onClick={submit}
        disabled={!url || busy}
        className="w-full btn-primary"
      >
        {busy ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-1.5 h-1.5 bg-slate-900 rounded-full animate-pulse" />
            <span className="w-1.5 h-1.5 bg-slate-900 rounded-full animate-pulse" style={{ animationDelay: "0.2s" }} />
            <span className="w-1.5 h-1.5 bg-slate-900 rounded-full animate-pulse" style={{ animationDelay: "0.4s" }} />
            Processing
          </span>
        ) : (
          <span className="flex items-center justify-center gap-2">
            Recommend Pattern
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </span>
        )}
      </button>
    </div>
  );
}
