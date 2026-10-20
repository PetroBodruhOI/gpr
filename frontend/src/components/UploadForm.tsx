import { useState } from "react";
import { predictFile } from "../api/client";
import { AlertIcon, ArrowRightIcon, CheckIcon, UploadIcon, XIcon } from "./icons";

interface Props {
  onStart: (taskId: string) => void;
}

// Keep in sync with MAX_UPLOAD_MB on the backend.
const MAX_UPLOAD_MB = 100;

const ALLOWED_TYPES = [
  "audio/wav",
  "audio/mpeg",
  "audio/mp3",
  "audio/ogg",
  "audio/flac",
  "audio/mp4",       // iPhone .m4a
  "audio/x-m4a",     // iPhone .m4a (альтернативний MIME)
  "audio/aac",
  "video/mp4",       // iPhone HEVC відео
  "video/quicktime", // iPhone .mov
];

export default function UploadForm({ onStart }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");

  const validateFile = (f: File | null): boolean => {
    if (!f) return false;

    if (!ALLOWED_TYPES.includes(f.type)) {
      setError("Підтримуються аудіофайли MP3, WAV, OGG, FLAC, M4A та відео MP4, MOV");
      setFile(null);
      return false;
    }

    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`Файл завеликий — максимум ${MAX_UPLOAD_MB} МБ`);
      setFile(null);
      return false;
    }

    setError("");
    setFile(f);
    return true;
  };

  const removeFile = () => {
    setFile(null);
    setError("");
  };

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const taskId = await predictFile(file);
      onStart(taskId);
    } catch {
      setError("Не вдалося завантажити файл. Спробуйте ще раз.");
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
    validateFile(e.dataTransfer.files?.[0] ?? null);
  };

  return (
    <div className="space-y-4">
      <label
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        data-drag={dragOver || undefined}
        className="group relative flex flex-col items-center justify-center w-full min-h-44 px-4 py-8 rounded-lg
                   cursor-pointer text-center border-[1.5px] border-dashed border-white/15 bg-ink/50
                   transition-colors duration-150
                   hover:border-glow/40 hover:bg-glow/[0.04]
                   data-[drag]:border-glow/70 data-[drag]:bg-glow/10
                   has-[:focus-visible]:outline has-[:focus-visible]:outline-2
                   has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent"
      >
        <span
          className="w-12 h-12 mb-3 rounded-lg flex items-center justify-center border border-glow/30
                     bg-glow/10 text-glow transition-transform duration-200 ease-out
                     group-data-[drag]:scale-110"
        >
          <UploadIcon className="w-6 h-6" />
        </span>

        <span className="text-white font-medium mb-1 [overflow-wrap:anywhere]">
          {dragOver
            ? "Відпустіть, щоб завантажити"
            : file
            ? file.name
            : "Перетягніть аудіо або відео або натисніть, щоб вибрати"}
        </span>
        <span className="text-white/45 text-xs">
          MP3 · WAV · OGG · FLAC · M4A · MP4 · MOV · до {MAX_UPLOAD_MB} МБ
        </span>

        <input
          type="file"
          accept=".mp3,.wav,.ogg,.flac,.m4a,.aac,.mp4,.mov,audio/*,video/*"
          disabled={busy}
          className="sr-only"
          onChange={(e) => validateFile(e.target.files?.[0] ?? null)}
        />
      </label>

      {error && (
        <div role="alert" className="flex items-start gap-2.5 p-3 rounded-lg text-sm text-red-200 bg-red-500/10 border border-red-500/30">
          <AlertIcon className="w-4 h-4 mt-0.5 flex-shrink-0" />
          {error}
        </div>
      )}

      {file && (
        <div data-file-card className="flex items-center gap-3 p-3 rounded-lg bg-glow/10 border border-glow/30">
          <span className="w-8 h-8 rounded-full bg-glow/20 text-glow flex items-center justify-center flex-shrink-0">
            <CheckIcon />
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-glow text-sm font-semibold">Файл готовий</p>
            <p className="text-white/60 text-sm truncate">{file.name}</p>
          </div>
          <button
            type="button"
            onClick={removeFile}
            aria-label="Прибрати файл"
            className="btn w-9 h-9 text-white/50 hover:text-red-300 hover:bg-white/5"
          >
            <XIcon className="w-5 h-5" />
          </button>
        </div>
      )}

      <button onClick={submit} disabled={!file || busy} className="btn-primary w-full">
        {busy ? (
          <>
            <span className="w-4 h-4 rounded-full border-2 border-ink/30 border-t-ink animate-spin" />
            Обробка
          </>
        ) : (
          <>
            Підібрати патерн
            <ArrowRightIcon />
          </>
        )}
      </button>
    </div>
  );
}
