import { useState } from "react";
import { predictFile } from "../api/client";

interface Props {
  onStart: (taskId: string) => void;
}

export default function UploadForm({ onStart }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const taskId = await predictFile(file);
      onStart(taskId);
    } finally {
      setBusy(false);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!busy) setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (busy) return;
    const dropped = e.dataTransfer.files?.[0];
    if (dropped && dropped.type.startsWith("audio/")) {
      setFile(dropped);
    } else if (dropped) {
      setFile(dropped);
    }
  };

  return (
    <div className="space-y-5">
      <div>
        <label className="block text-xs font-semibold text-white/50 mb-2 uppercase tracking-wider">
          Аудіофайл
        </label>
        <label
          onDragOver={handleDragOver}
          onDragEnter={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className="group flex flex-col items-center justify-center w-full h-44 rounded-lg cursor-pointer transition-all duration-200 relative overflow-hidden"
          style={{
            background: dragOver ? "rgba(16, 185, 129, 0.1)" : "rgba(10, 20, 36, 0.4)",
            border: dragOver
              ? "1.5px dashed rgba(16, 185, 129, 0.6)"
              : "1.5px dashed rgba(255, 255, 255, 0.15)",
          }}
        >
          <div className={`absolute inset-0 bg-emerald-500/5 transition-opacity ${dragOver ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`} />
          <div className="relative flex flex-col items-center justify-center pointer-events-none">
            <div className={`w-12 h-12 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center mb-3 transition-all ${dragOver ? "bg-emerald-500/30 scale-110" : "group-hover:bg-emerald-500/20"}`}>
              <svg className="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <p className="text-white font-medium mb-1">
              {dragOver ? "Відпустіть, щоб завантажити" : file ? file.name : "Перетягніть аудіофайл або натисніть, щоб вибрати"}
            </p>
            <p className="text-white/40 text-xs">MP3 · WAV · OGG · FLAC</p>
          </div>
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            disabled={busy}
            className="hidden"
          />
        </label>
      </div>

      {file && (
        <div className="p-4 rounded-xl flex items-center gap-3"
             style={{
               background: "rgba(34, 197, 94, 0.1)",
               border: "1px solid rgba(34, 197, 94, 0.3)"
             }}>
          <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-green-300 text-sm font-semibold">Файл готовий</p>
            <p className="text-white/60 text-sm truncate">{file.name}</p>
          </div>
        </div>
      )}

      <button
        onClick={submit}
        disabled={!file || busy}
        className="w-full btn-primary"
      >
        {busy ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-1.5 h-1.5 bg-slate-900 rounded-full animate-pulse" />
            <span className="w-1.5 h-1.5 bg-slate-900 rounded-full animate-pulse" style={{ animationDelay: "0.2s" }} />
            <span className="w-1.5 h-1.5 bg-slate-900 rounded-full animate-pulse" style={{ animationDelay: "0.4s" }} />
            Обробка
          </span>
        ) : (
          <span className="flex items-center justify-center gap-2">
            Підібрати патерн
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M17 8l4 4m0 0l-4 4m4-4H3" />
            </svg>
          </span>
        )}
      </button>
    </div>
  );
}
