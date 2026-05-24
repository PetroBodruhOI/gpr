import { useState } from "react";
import UploadForm from "./components/UploadForm";
import UrlForm from "./components/UrlForm";
import ProgressBar from "./components/ProgressBar";
import ResultCard from "./components/ResultCard";
import PatternsGallery from "./components/PatternsGallery";
import { TaskStatus, pollTask } from "./api/client";

type Mode = "file" | "url";
type View = "patterns" | "analyze";

export default function App() {
  const [view, setView] = useState<View>("analyze");
  const [mode, setMode] = useState<Mode>("url");
  const [task, setTask] = useState<TaskStatus | null>(null);
  const [error, setError] = useState<string>("");

  const handleTaskStart = async (taskId: string) => {
    setError("");
    setTask({ task_id: taskId, status: "pending", progress: 0, message: "" });
    try {
      const finalTask = await pollTask(taskId, (t) => setTask(t));
      setTask(finalTask);
    } catch (e) {
      setError(String(e));
    }
  };

  const resetTask = () => {
    setTask(null);
    setError("");
  };

  const switchView = (v: View) => {
    setView(v);
    if (v === "patterns") resetTask();
  };

  const navTabClass = (active: boolean) =>
    `relative px-5 py-3 text-sm md:text-base font-semibold tracking-wide transition-all duration-200 ${
      active
        ? "text-white"
        : "text-white/55 hover:text-white/85"
    }`;

  const isTaskRunning = task !== null;
  const isTaskDone = task?.status === "done" || task?.status === "error";

  // Wider container for patterns view so they don't get cramped
  const containerWidth = view === "patterns" ? "max-w-7xl" : "max-w-4xl";

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Subtle background accents */}
      <div className="fixed -top-40 -right-40 w-96 h-96 bg-blue-700/15 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed -bottom-40 -left-40 w-96 h-96 bg-emerald-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Top header bar */}
      <header className="relative border-b border-white/5 backdrop-blur-md bg-slate-900/40 z-10">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-3 py-4">
            <div className="font-bold text-base tracking-widest text-yellow-300">GPR</div>
            <span className="hidden sm:inline text-white/30 text-xs uppercase tracking-wider">
              Guitar Pattern Recommender
            </span>
          </div>
          <nav className="flex h-full">
            <button
              onClick={() => switchView("analyze")}
              className={navTabClass(view === "analyze")}
            >
              Аналіз
              {view === "analyze" && (
                <span className="absolute left-2 right-2 bottom-0 h-0.5 bg-yellow-400 rounded-t-full" />
              )}
            </button>
            <button
              onClick={() => switchView("patterns")}
              className={navTabClass(view === "patterns")}
            >
              Список патернів
              {view === "patterns" && (
                <span className="absolute left-2 right-2 bottom-0 h-0.5 bg-yellow-400 rounded-t-full" />
              )}
            </button>
          </nav>
        </div>
      </header>

      <div className={`relative ${containerWidth} mx-auto py-12 px-4`}>
        {/* Title block */}
        <div className="mb-10 text-center">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-5 leading-tight tracking-tight">
            Guitar Pattern
            <br />
            <span className="gradient-text">Recommender</span>
          </h1>
          <p className="text-white/60 text-base max-w-xl mx-auto leading-relaxed">
            Не існує єдиного правильного способу зіграти музичний твір.
          </p>
          <p className="text-white/60 text-base max-w-xl mx-auto leading-relaxed">
            GPR порекомендує патерн, який підійде до вашого треку — або один із найбільш доречних варіантів.
          </p>
        </div>

        {view === "patterns" && <PatternsGallery />}

        {view === "analyze" && (
          <>
            {/* Form section — hidden once a task is in flight */}
            {!isTaskRunning && (
              <>
                <div className="flex gap-1.5 mb-6 p-1.5 rounded-xl bg-slate-900/40 backdrop-blur-xl border border-white/5">
                  <button
                    onClick={() => setMode("url")}
                    className={`flex-1 py-2.5 px-4 rounded-lg font-medium transition-all duration-200 ${
                      mode === "url"
                        ? "bg-blue-900/60 text-white border border-blue-500/30"
                        : "text-white/50 hover:text-white/80"
                    }`}
                  >
                    Посилання на YouTube
                  </button>
                  <button
                    onClick={() => setMode("file")}
                    className={`flex-1 py-2.5 px-4 rounded-lg font-medium transition-all duration-200 ${
                      mode === "file"
                        ? "bg-blue-900/60 text-white border border-blue-500/30"
                        : "text-white/50 hover:text-white/80"
                    }`}
                  >
                    Завантажити файл
                  </button>
                </div>
                <div className="card mb-6">
                  {mode === "url" && <UrlForm onStart={handleTaskStart} />}
                  {mode === "file" && <UploadForm onStart={handleTaskStart} />}
                </div>
              </>
            )}

            {/* Progress while task runs */}
            {task && <ProgressBar progress={task.progress} message={task.message} />}

            {/* Result */}
            {task?.status === "done" && task.result && (
              <ResultCard result={task.result} taskId={task.task_id} />
            )}

            {/* Errors */}
            {task?.status === "error" && (
              <div className="card mt-6" style={{ borderColor: "rgba(239, 68, 68, 0.4)" }}>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
                    <span className="text-xl">⚠</span>
                  </div>
                  <div>
                    <h3 className="font-semibold text-red-300">Помилка</h3>
                    <p className="text-white/60 text-sm">{task.message}</p>
                  </div>
                </div>
              </div>
            )}
            {error && (
              <div className="card mt-6" style={{ borderColor: "rgba(239, 68, 68, 0.4)" }}>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
                    <span className="text-xl">⚠</span>
                  </div>
                  <p className="text-white/80 font-medium">{error}</p>
                </div>
              </div>
            )}

            {/* Reset button once the task is finished */}
            {isTaskDone && (
              <div className="mt-6 flex justify-center">
                <button onClick={resetTask} className="btn-secondary">
                  ← Аналізувати інший трек
                </button>
              </div>
            )}
          </>
        )}

        {/* Footer */}
        <div className="mt-16 text-center text-white/30 text-xs">
          GPR · Guitar Pattern Recommender
        </div>
      </div>
    </div>
  );
}
