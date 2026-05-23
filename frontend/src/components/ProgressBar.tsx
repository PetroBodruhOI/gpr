interface Props {
  progress: number;
  message: string;
}

export default function ProgressBar({ progress, message }: Props) {
  return (
    <div className="card mb-6 relative overflow-hidden">
      <div className="relative">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/15 flex items-center justify-center border border-emerald-500/30">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            </div>
            <h3 className="font-medium text-white">Processing</h3>
          </div>
          <span className="text-xl font-semibold text-yellow-300">{progress}%</span>
        </div>
        <div className="mb-3 bg-slate-900/60 rounded-full h-1.5 overflow-hidden border border-white/5">
          <div
            style={{ width: `${progress}%` }}
            className="h-full bg-emerald-500 transition-all duration-500"
          />
        </div>
        <p className="text-white/55 text-sm">{message}</p>
      </div>
    </div>
  );
}
