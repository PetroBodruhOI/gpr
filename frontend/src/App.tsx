import { useRef, useState } from "react";
import UploadForm from "./components/UploadForm";
import ProgressBar from "./components/ProgressBar";
import ResultCard from "./components/ResultCard";
import PatternsGallery from "./components/PatternsGallery";
import { TaskStatus, pollTask } from "./api/client";

type View = "patterns" | "analyze";

export default function App() {
  const [view, setView] = useState<View>("analyze");
  const [task, setTask] = useState<TaskStatus | null>(null);
  const [error, setError] = useState<string>("");
  const formRef = useRef<HTMLDivElement | null>(null);

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

  const scrollToForm = () => {
    // wait one tick so the form is mounted after view switch / task reset
    requestAnimationFrame(() => {
      formRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  const switchView = (v: View) => {
    setView(v);
    if (v === "patterns") resetTask();
  };

  // CTA "Аналіз": always returns to the recommender form
  const goAnalyze = () => {
    setView("analyze");
    resetTask();
    scrollToForm();
  };

  const isTaskRunning = task !== null;
  const isTaskDone = task?.status === "done" || task?.status === "error";

  // Wider container for patterns view so they don't get cramped
  const containerWidth = view === "patterns" ? "max-w-7xl" : "max-w-4xl";

  return (
    <div className="min-h-screen relative">
      {/* Subtle background accents */}
      <div className="fixed -top-40 -right-40 w-96 h-96 bg-blue-700/15 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed -bottom-40 -left-40 w-96 h-96 bg-emerald-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Top header bar — sticky, always visible */}
      <header className="sticky top-0 border-b border-white/5 backdrop-blur-xl bg-slate-900/80 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between gap-3">
          <button
            onClick={goAnalyze}
            className="flex items-center gap-3 py-4 cursor-pointer group"
          >
            <div className="font-bold text-base tracking-widest text-yellow-300 group-hover:text-yellow-200 transition-colors">
              GPR
            </div>
            <span className="hidden sm:inline text-white/30 text-xs uppercase tracking-wider group-hover:text-white/50 transition-colors">
              Guitar Pattern Recommender
            </span>
          </button>
          <nav className="flex items-center gap-2 sm:gap-3">
            <button
              onClick={() => switchView("patterns")}
              className={`relative px-3 sm:px-4 py-2 text-sm font-semibold transition-all duration-200 ${
                view === "patterns"
                  ? "text-white"
                  : "text-white/55 hover:text-white/85"
              }`}
            >
              Список патернів
              {view === "patterns" && (
                <span className="absolute left-2 right-2 -bottom-0.5 h-0.5 bg-yellow-400 rounded-t-full" />
              )}
            </button>
            <button
              onClick={goAnalyze}
              className="px-4 sm:px-5 py-2 rounded-lg font-semibold text-sm text-slate-900 transition-all duration-200 flex items-center gap-1.5"
              style={{ background: "#facc15", border: "1px solid #eab308" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "#fde047")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "#facc15")}
            >
              Аналіз
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2.5}
                  d="M3 12h2l2-7 4 14 3-10 2 6h5"
                />
              </svg>
            </button>
          </nav>
        </div>
      </header>

      <div className={`relative ${containerWidth} mx-auto py-6 sm:py-12 px-4`}>
        {/* Title block */}
        <div className="mb-6 sm:mb-10 text-center">
          <h1 className="text-3xl sm:text-5xl md:text-6xl font-bold text-white mb-4 sm:mb-5 leading-tight tracking-tight">
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
              <div ref={formRef} className="scroll-mt-24">
                <div className="card mb-6">
                  {<UploadForm onStart={handleTaskStart} />}
                </div>
              </div>
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
