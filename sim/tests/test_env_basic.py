import numpy as np
from sim.envs.multi_drone_quad_env import MultiDroneQuadEnv

env = MultiDroneQuadEnv(num_drones=5, gui=False, max_steps=200)
obs, _ = env.reset(options=dict(
    leader_speed_scale=0.0, spawn_in_formation=True, disable_dynamic=True,
    formation_spacing=0.9, min_sep=0.55, sep_gain=1.2, sep_hysteresis=0.06,
    max_roll_deg=5, max_pitch_deg=5, thrust_delta_scale=0.25
))
for _ in range(10):
    a = np.zeros(env.num_drones * env.act_per_drone, dtype=np.float32)
    obs, r, term, trunc, info = env.step(a)
    assert np.isfinite(r)
env.close()
print("env smoke test ok")
