# sim/envs/callbacks.py
from typing import Any, Dict, List
from stable_baselines3.common.callbacks import BaseCallback


class CustomMetricsCallback(BaseCallback):
    """
    Logs custom env metrics that are stored on each env as `env.last_metrics`
    (a dict of scalar values).

    Works with both DummyVecEnv and SubprocVecEnv.
    """

    def __init__(self, log_freq: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Only log every N callback calls
        if self.n_calls % self.log_freq != 0:
            return True

        vec_env = self.training_env

        # ---- grab per-env last_metrics dicts ----
        metrics_list: List[Dict[str, Any]] = []

        # DummyVecEnv has `.envs`
        if hasattr(vec_env, "envs"):
            for env in vec_env.envs:
                m = getattr(env, "last_metrics", None)
                if isinstance(m, dict) and m:
                    metrics_list.append(m)
        else:
            # SubprocVecEnv: use get_attr across workers
            try:
                attrs = vec_env.get_attr("last_metrics")
                for m in attrs:
                    if isinstance(m, dict) and m:
                        metrics_list.append(m)
            except Exception:
                # If anything goes wrong, just skip logging this step
                return True

        if not metrics_list:
            # nothing to log
            return True

        # ---- average each metric across envs ----
        keys = metrics_list[0].keys()
        for k in keys:
            vals = [float(m.get(k, 0.0)) for m in metrics_list if k in m]
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            self.logger.record(f"custom/{k}", mean_val)

        return True
