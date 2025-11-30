from typing import Dict, Any, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EnvMetricsLogger(BaseCallback):
    """
    Logs env-provided metrics to TensorBoard during training.
    Looks for 'metrics' in each info dict (per-env) and aggregates per iteration.
    """
    def __init__(self, prefix: str = "env", verbose: int = 0):
        super().__init__(verbose)
        self.prefix = prefix
        self.buf: Dict[str, List[float]] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            m: Dict[str, Any] = info.get("metrics", {})
            if not isinstance(m, dict):
                continue
            for k, v in m.items():
                try:
                    val = float(v)
                except Exception:
                    continue
                self.buf.setdefault(k, []).append(val)
        return True

    def _on_rollout_end(self) -> None:
        for k, vals in self.buf.items():
            arr = np.asarray(vals, dtype=np.float32)
            if arr.size == 0:
                continue
            self.logger.record(f"{self.prefix}/{k}_mean", float(np.nanmean(arr)))
            self.logger.record(f"{self.prefix}/{k}_p90", float(np.nanpercentile(arr, 90)))
            self.logger.record(f"{self.prefix}/{k}_min", float(np.nanmin(arr)))
        self.buf.clear()
