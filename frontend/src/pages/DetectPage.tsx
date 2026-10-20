import UploadForm from "../components/UploadForm";
import ProgressBar from "../components/ProgressBar";
import ResultCard from "../components/ResultCard";
import { AlertIcon, ArrowLeftIcon } from "../components/icons";
import type { TaskStatus } from "../api/client";

function ErrorCard({ title, message, onRetry }: {
  title: string;
  message: string;
  onRetry: () => void;
}) {
  return (
    <div role="alert" className="card enter border-red-400/50">
      <div className="flex items-start gap-3">
        <AlertIcon className="w-5 h-5 mt-1 text-red-300 flex-shrink-0" />
        <div className="min-w-0 flex-1">
          <h3 className="text-lg font-bold text-red-200">{title}</h3>
          <p className="text-white/70 text-sm mt-1 break-words">{message}</p>
          <button onClick={onRetry} className="btn-secondary mt-4 text-sm">
            Спробувати інший файл
          </button>
        </div>
      </div>
    </div>
  );
}

interface Props {
  task: TaskStatus | null;
  error: string;
  onStart: (taskId: string) => void;
  onReset: () => void;
}

export default function DetectPage({ task, error, onStart, onReset }: Props) {
  const isIdle = task === null && !error;
  const isRunning = task !== null && task.status !== "done" && task.status !== "error" && !error;

  return (
    <div className="max-w-3xl mx-auto">
      {isIdle && (
        <>
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight">Визначити патерн</h1>
          <p className="mt-3 mb-8 text-lg text-white/70 leading-relaxed">
            Завантажте аудіо або відео з гітарою — GPR підбере бій чи арпеджіо і покаже, як його грати.
          </p>
          <div className="card">
            <UploadForm onStart={onStart} />
          </div>
        </>
      )}

      {isRunning && task && <ProgressBar progress={task.progress} message={task.message} />}

      {task?.status === "done" && task.result && (
        <ResultCard result={task.result} taskId={task.task_id} onReset={onReset} />
      )}

      {task?.status === "error" && (
        <ErrorCard title="Не вдалося проаналізувати трек" message={task.message} onRetry={onReset} />
      )}
      {error && <ErrorCard title="Втрачено зв'язок із сервером" message={error} onRetry={onReset} />}

      {task?.status === "done" && (
        <div className="mt-8 flex justify-center">
          <button onClick={onReset} className="btn-secondary">
            <ArrowLeftIcon />
            Аналізувати інший трек
          </button>
        </div>
      )}
    </div>
  );
}
