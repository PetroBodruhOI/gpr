import { useState } from "react";
import HomePage from "./pages/HomePage";
import DetectPage from "./pages/DetectPage";
import PatternsGallery from "./components/PatternsGallery";
import { TaskStatus, pollTask } from "./api/client";
import { href, Route, useHashRoute } from "./useHashRoute";

function NavLink({ to, current, children }: { to: Route; current: Route; children: React.ReactNode }) {
  const active = to === current;
  return (
    <a
      href={href(to)}
      aria-current={active ? "page" : undefined}
      className={`px-3 py-2 text-sm sm:text-base font-semibold transition-colors duration-150 ${
        active ? "text-white" : "text-white/60 hover:text-white"
      }`}
    >
      {children}
    </a>
  );
}

export default function App() {
  const [route, navigate] = useHashRoute();
  const [task, setTask] = useState<TaskStatus | null>(null);
  const [error, setError] = useState("");

  const handleTaskStart = async (taskId: string) => {
    setError("");
    setTask({ task_id: taskId, status: "pending", progress: 0, message: "" });
    navigate("detect");
    try {
      setTask(await pollTask(taskId, setTask));
    } catch (e) {
      setError(String(e));
    }
  };

  const resetTask = () => {
    setTask(null);
    setError("");
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-50 bg-ink/90 backdrop-blur-md border-b border-white/[0.06]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-3">
          <a href={href("home")} className="flex items-baseline gap-3 group">
            <span className="font-display font-extrabold text-2xl text-glow">GPR</span>
            <span className="hidden md:inline text-white/50 text-sm transition-colors group-hover:text-white/75">
              Guitar Pattern Recommender
            </span>
          </a>
          <nav className="flex items-center gap-1 sm:gap-2">
            <NavLink to="home" current={route}>Головна</NavLink>
            <NavLink to="patterns" current={route}>Патерни</NavLink>
            <a
              href={href("detect")}
              aria-current={route === "detect" ? "page" : undefined}
              className="btn-primary ml-1 px-4 py-2 text-sm sm:text-base"
            >
              <span className="sm:hidden">Визначити</span>
              <span className="hidden sm:inline">Визначити патерн</span>
            </a>
          </nav>
        </div>
      </header>

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 py-10 sm:py-14">
        {route === "home" && <HomePage onStart={handleTaskStart} />}
        {route === "patterns" && <PatternsGallery />}
        {route === "detect" && (
          <DetectPage task={task} error={error} onStart={handleTaskStart} onReset={resetTask} />
        )}
      </main>

      <footer className="mt-16 border-t border-white/[0.06] py-8 text-center text-white/45 text-sm">
        GPR · Guitar Pattern Recommender
      </footer>
    </div>
  );
}
