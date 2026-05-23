from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict

class PredictUrlRequest(BaseModel):
    url: HttpUrl
    start_sec:    Optional[float] = None
    duration_sec: Optional[float] = None

class ChunkResult(BaseModel):
    chunk_idx:  int
    time_start: float
    time_end:   float
    label:      str
    confidence: float
    probs:      Dict[str, float]  # {"6a": 0.02, "8b": 0.91, ...}

class TaskStatus(BaseModel):
    task_id:  str
    status:   str          # pending | processing | done | error
    progress: int = 0      # 0–100
    message:  str = ""
    result:   Optional[dict] = None
