import os
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import pybullet_data

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets", "crazyflie", "cf_assets")
URDF_PATH = os.path.join(ASSETS_DIR, "cf2x.urdf")


class MultiDroneQuadEnv(gym.Env):
    """
    Centralized multi-drone quadrotor environment with high-level attitude control.
    Eval toggles (reset(options=...)):
      - leader_speed_scale: float (0.0 freezes leader trajectory motion)
      - spawn_in_formation: bool (spawn at desired diamond slots)
      - disable_dynamic:    bool (no chasing sphere)
      - debug_diamond:      bool (render desired slots as green stems in GUI)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        num_drones: int = 5,
        gui: bool = False,
        max_steps: int = 2000,
        render_mode: str | None = None,):
        super().__init__()
        
        if num_drones < 1:
            raise ValueError("MultiDroneQuadEnv requires at least 1 drone.")
        self.num_drones = num_drones
        self.gui = gui
        
        self.render_mode = render_mode
        if render_mode == "human":
            gui = True
        if render_mode == "rgb_array":
            gui = False
        self.gui = gui

        if p.isConnected():
            self.physics_client = p.getConnectionInfo().get("physicsClientId", 0)
        else:
            self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)

        p.setRealTimeSimulation(0)
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.collision_happened = False
        self.last_metrics = {}
        self.leader_traj_t = 0.0
        self.drone_ids: list[int] = []
        self.obstacle_ids: list[int] = []
        self.dynamic_obstacle_id: int | None = None
        self.dynamic_phase = 0.0
        self._last_retarget_time = 0.0
        self.act_per_drone = 4
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_drones * self.act_per_drone,), dtype=np.float32
        )
        self._prev_action = np.zeros((self.num_drones, self.act_per_drone), dtype=np.float32)
        self.last_action = np.zeros_like(self._prev_action)
        self.last_action_delta = np.zeros_like(self._prev_action)

        self.self_dim = 12
        self.neighbor_dim = 6 * (self.num_drones - 1)
        self.num_static_obstacles = 4
        self.static_obs_dim = 3 * self.num_static_obstacles
        self.dynamic_obs_dim = 3
        self.leader_err_dim = 3
        self.per_drone_obs_dim = (
            self.self_dim + self.neighbor_dim + self.static_obs_dim
            + self.dynamic_obs_dim + self.leader_err_dim
        )
        global_obs_dim = self.per_drone_obs_dim * self.num_drones
        obs_high = np.ones(global_obs_dim, dtype=np.float32) * 1e6
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, shape=(global_obs_dim,), dtype=np.float32
        )

        self.time_step = 1.0 / 240.0
        self.target_z = 1.0
        self.mass = None
        self.hover_thrust = None

        self.max_roll = np.deg2rad(10.0)
        self.max_pitch = np.deg2rad(10.0)
        self.max_yaw_rate = np.deg2rad(60.0)

        self.kp_roll = 4.0;  self.kd_roll = 0.5
        self.kp_pitch = 4.0; self.kd_pitch = 0.5
        self.kp_yaw_rate = 0.3; self.kd_yaw = 0.05
        self.kp_z = 25.0
        self.kd_z = 10.0


        self.max_torque = np.array([1e-3, 1e-3, 5e-4], dtype=np.float32)
        self.thrust_delta_scale = 0.4

        self.formation_layout = "diamond"

        self.cfg = dict(
            leader_speed_scale = 0.3,
            spawn_in_formation = True,
            disable_dynamic    = True,
            debug_diamond      = False,
            debug_separation   = False,
            # --- stability / anti-chatter shaping ---
            alive_bonus = 0.0,                 # + per step (raw units, before reward_scale)
            formation_break_penalty = 0.0,     # terminal penalty on formation-break (raw units)

            att_w = 0.5,                       # attitude penalty weight (was hardcoded via getattr)
            ang_vel_w = 0.0,                   # penalize |omega| (body/world ang vel magnitude)

            action_mag_w = 0.0,                # penalize |action|^2
            action_rate_w = 0.0,               # penalize |Δaction|^2

            action_smooth_alpha = 0.0,         # 0..0.95 low-pass on commands (env-side)

            form_w_mean = 1.5,
            form_w_max  = 0.4,
            form_var_gain = 0.3,
            huber_delta   = 0.4,

            alt_w = 0.4,
            speed_smooth_gain = 0.01,

            formation_spacing = 0.8,
            min_sep = 0.5,
            sep_radius = 0.35,
            sep_gain = 1.5,
            sep_hysteresis = 0.05,
            sep_force = 0.8,

            static_clear_gain = 0.4,

            threat_radius = 1.2,
            danger_radius = 0.8,
            safe_radius   = 1.6,
            evade_gain    = 0.5,
            avoid_scale   = 3.0,
            safe_bonus    = 1.0,
            form_under_threat_gain = 1.4,

            terminate_on_drone_collision = True,
            collision_penalty = 8.0,

            chase_mode = "nearest",
            retarget_interval = 1.5,
            retarget_on_close = True,
            retarget_close_dist = 0.5,
            dynamic_aggression = 0.7,
            debug_target = False,
            thrust_delta_scale = 0.4,
            max_roll_deg = 10.0,
            max_pitch_deg = 10.0,
            max_yaw_rate_deg = 60.0,

            workspace_radius = 2.0,
            workspace_gain   = 0.4,
            tilt_terminate_deg = 30.0, # if |roll| or |pitch| > this, end the episode

            # formation-break termination thresholds (meters)
            formation_break_mean = 2.0,  # terminate if mean formation error exceeds this
            formation_break_max  = 3.0,  # terminate if any drone exceeds this

            # soft physical repulsion from obstacles / dynamic sphere
            obstacle_repulse_radius = 0.7,
            obstacle_repulse_gain   = 1.0,
            dynamic_repulse_radius  = 0.9,
            dynamic_repulse_gain    = 1.2,

            # --- STL-style reward options ---
            use_stl_reward  = False,   # when True, reward = STL-style robustness margin
            stl_form_tol    = 0.5,     # desired upper bound on mean formation error (m)
            stl_max_tol     = 1.0,     # desired upper bound on max formation error (m)
            stl_reward_clip = 2.0,     # clamp STL robustness in [-clip, clip]

            # scale for legacy shaped reward (kept for backward compatibility)
            reward_scale    = 0.01,
        )


        self._apply_cfg()
        

        if self.formation_layout == "diamond":
            r = float(self.cfg["formation_spacing"])
            base_offsets = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
            for a_deg in (0.0, 90.0, 180.0, 270.0):
                a = np.deg2rad(a_deg)
                base_offsets.append(np.array([r * np.cos(a), r * np.sin(a), 0.0], dtype=np.float32))
        else:
            base_offsets = [
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.6, -0.4, 0.0], dtype=np.float32),
                np.array([0.6,  0.4, 0.0], dtype=np.float32),
                np.array([1.2, -0.3, 0.0], dtype=np.float32),
                np.array([1.2,  0.3, 0.0], dtype=np.float32),
            ]

        if self.num_drones > len(base_offsets):
            raise ValueError(
                f"Formation offsets only defined for up to {len(base_offsets)} drones; "
                f"got num_drones={self.num_drones}."
            )
        self.formation_offsets = {i: base_offsets[i] for i in range(self.num_drones)}

        def _min_pairwise_distance(offsets):
            dmin = float('inf')
            for i in range(len(offsets)):
                for j in range(i+1, len(offsets)):
                    dmin = min(dmin, float(np.linalg.norm(offsets[i] - offsets[j])))
            return dmin

        min_pair = _min_pairwise_distance([self.formation_offsets[i] for i in range(self.num_drones)])
        if min_pair < self.cfg["min_sep"] + 0.05:
            print(
                f"[WARN] formation spacing {min_pair:.2f} < min_sep+margin {self.cfg['min_sep']+0.05:.2f}. "
                "Expect separation/collision chatter. Consider spacing↑ or min_sep↓."
            )

    def _apply_cfg(self):
        """Mirror cfg entries to attributes that other methods expect."""
        for k, v in self.cfg.items():
            setattr(self, k, v)

        self.max_roll = np.deg2rad(float(self.cfg.get("max_roll_deg", 10.0)))
        self.max_pitch = np.deg2rad(float(self.cfg.get("max_pitch_deg", 10.0)))
        self.max_yaw_rate = np.deg2rad(float(self.cfg.get("max_yaw_rate_deg", 60.0)))

            
    @staticmethod
    def _huber(x: np.ndarray, delta: float) -> np.ndarray:
        """Smooth L1; robust to outliers."""
        a = np.abs(x)
        return np.where(a <= delta, 0.5 * (x ** 2), delta * (a - 0.5 * delta))


    def leader_trajectory(self, t: float) -> np.ndarray:
        s = float(self.leader_speed_scale)
        x = 0.3 * s * t
        y = 0.4 * np.sin(0.4 * s * t)
        z = self.target_z
        return np.array([x, y, z], dtype=np.float32)


    def reset(self, *, seed=None, options=None):
        
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        opts = options or {}

        if opts:
            self.cfg.update({k: v for k, v in opts.items() if k in self.cfg})
        self._apply_cfg()
        
        def _deg_override(opts_key: str, attr_name: str):
            val = None
            if opts_key in opts and opts[opts_key] is not None:
                val = opts[opts_key]
            elif isinstance(self.cfg, dict) and self.cfg.get(opts_key) is not None:
                val = self.cfg[opts_key]
            if val is not None:
                setattr(self, attr_name, np.deg2rad(float(val)))

        _deg_override("max_roll_deg",       "max_roll")
        _deg_override("max_pitch_deg",      "max_pitch")
        _deg_override("max_yaw_rate_deg",   "max_yaw_rate")

        if "thrust_delta_scale" in opts and opts["thrust_delta_scale"] is not None:
            self.thrust_delta_scale = float(opts["thrust_delta_scale"])
        elif isinstance(self.cfg, dict) and self.cfg.get("thrust_delta_scale") is not None:
            self.thrust_delta_scale = float(self.cfg["thrust_delta_scale"])


        if "formation_spacing" in opts:
            if self.formation_layout == "diamond":
                r = float(self.cfg["formation_spacing"])
                base_offsets = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
                for a_deg in (0.0, 90.0, 180.0, 270.0):
                    a = np.deg2rad(a_deg)
                    base_offsets.append(np.array([r * np.cos(a), r * np.sin(a), 0.0], dtype=np.float32))
            else:
                base_offsets = [
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.6, -0.4, 0.0], dtype=np.float32),
                    np.array([0.6,  0.4, 0.0], dtype=np.float32),
                    np.array([1.2, -0.3, 0.0], dtype=np.float32),
                    np.array([1.2,  0.3, 0.0], dtype=np.float32),
                ]
            self.formation_offsets = {i: base_offsets[i] for i in range(self.num_drones)}

        self.collision_happened = False
        self.step_count = 0
        self.leader_traj_t = 0.0
        self.dynamic_phase = 0.0
        self._last_retarget_time = 0.0
        self._prev_action.fill(0.0)
        self.last_action.fill(0.0)
        self.last_action_delta.fill(0.0)



        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        self.drone_ids = self._spawn_drones(self.num_drones)
        self.obstacle_ids = self._spawn_static_obstacles()

        if self.disable_dynamic:
            self.dynamic_obstacle_id = None
        else:
            self.dynamic_obstacle_id = self._spawn_dynamic_obstacle()

        if "pos_jitter" in opts:
            for i in range(self.num_drones):
                base_pos, base_ori = p.getBasePositionAndOrientation(self.drone_ids[i])
                jitter = np.array(opts["pos_jitter"][i], dtype=np.float32)
                p.resetBasePositionAndOrientation(self.drone_ids[i], (np.array(base_pos) + jitter).tolist(), base_ori)

        if "yaw_jitter" in opts:
            for i in range(self.num_drones):
                yaw = float(opts["yaw_jitter"][i])
                pos, _ = p.getBasePositionAndOrientation(self.drone_ids[i])
                p.resetBasePositionAndOrientation(self.drone_ids[i], pos, p.getQuaternionFromEuler([0, 0, yaw]))

        if "vel_jitter" in opts:
            for i in range(self.num_drones):
                vel = np.array(opts["vel_jitter"][i], dtype=np.float32)
                p.resetBaseVelocity(self.drone_ids[i], vel.tolist(), [0, 0, 0])

        if "obstacle_jitter" in opts:
            jitter = np.array(opts["obstacle_jitter"], dtype=np.float32)
            for oid in self.obstacle_ids:
                pos, ori = p.getBasePositionAndOrientation(oid)
                p.resetBasePositionAndOrientation(oid, (np.array(pos) + jitter).tolist(), ori)

        if "dynamic_jitter" in opts and self.dynamic_obstacle_id is not None:
            pos, ori = p.getBasePositionAndOrientation(self.dynamic_obstacle_id)
            p.resetBasePositionAndOrientation(
                self.dynamic_obstacle_id,
                (np.array(pos) + np.array(opts["dynamic_jitter"], dtype=np.float32)).tolist(),
                ori,
            )

        dyn_info = p.getDynamicsInfo(self.drone_ids[0], -1)
        self.mass = dyn_info[0]
        self.hover_thrust = self.mass * 9.81

        calib_steps = 180
        z0 = None
        for s in range(calib_steps):
            for i in range(self.num_drones):
                p.applyExternalForce(
                    self.drone_ids[i], -1, [0.0, 0.0, self.hover_thrust], [0.0, 0.0, 0.0], p.WORLD_FRAME
                )
            p.stepSimulation()
            if s == 0:
                pos0, _ = p.getBasePositionAndOrientation(self.drone_ids[0]); z0 = pos0[2]
            elif s == calib_steps - 1:
                posN, _ = p.getBasePositionAndOrientation(self.drone_ids[0])
                drift = posN[2] - z0
                T = calib_steps * self.time_step
                F_correction = 2.0 * self.mass * drift / (T * T)
                self.hover_thrust = max(self.hover_thrust - F_correction, 0.01)

        quat0 = p.getQuaternionFromEuler([0, 0, 0])
        if self.spawn_in_formation:
            leader_des = self.leader_trajectory(0.0)
            for i in range(self.num_drones):
                start_pos = leader_des + self.formation_offsets[i]
                p.resetBasePositionAndOrientation(self.drone_ids[i], start_pos.tolist(), quat0)
                p.resetBaseVelocity(self.drone_ids[i], [0, 0, 0], [0, 0, 0])
        else:
            for i in range(self.num_drones):
                start_pos = np.array([i * 0.3, 0.0, self.target_z], dtype=np.float32)
                p.resetBasePositionAndOrientation(self.drone_ids[i], start_pos, quat0)
                p.resetBaseVelocity(self.drone_ids[i], [0, 0, 0], [0, 0, 0])

        self.chase_target_drone = self._pick_chase_target(self.chase_mode)

        warmup_steps = 240
        zero_action = np.zeros((self.num_drones, self.act_per_drone), dtype=np.float32)
        for _ in range(warmup_steps):
            for i in range(self.num_drones):
                self._apply_action(i, zero_action[i])
            p.stepSimulation()

        fine_keys = (
            "form_w_mean","form_w_max","form_var_gain","form_under_threat_gain","huber_delta",
            "alt_w","speed_smooth_gain","sep_radius","sep_gain",
            "static_clear_gain","threat_radius","evade_gain",
            "danger_radius","safe_radius","avoid_scale","safe_bonus","collision_penalty",
            "thrust_delta_scale","max_roll_deg","max_pitch_deg","max_yaw_rate_deg",
            "reward_scale","stl_form_tol","stl_max_tol","stl_reward_clip",
        )

        for k in fine_keys:
            if k in opts:
                setattr(self, k, float(opts[k]))

        obs_all = self._get_all_obs()
        return obs_all.flatten().astype(np.float32), {}

    
    def _apply_separation_forces(self):
        """
        Repel pairs closer than `min_sep` with a smooth radial push + damping.
        Math:
        r0 = min_sep          (start of keep-out)
        r1 = min_sep - hys    (full-strength region)
        s  = smoothstep((r0 - d)/(r0 - r1)) in [0,1]  (C¹ ramp)
        F  = (+k * s) along line of centers  -  (c * v_rel_along_line)
        """
        n = self.num_drones
        if n <= 1:
            return

        r0 = float(self.min_sep)
        r1 = max(0.05, r0 - float(self.sep_hysteresis))

        pos = np.empty((n, 3), dtype=np.float32)
        vel = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            p_i, _ = p.getBasePositionAndOrientation(self.drone_ids[i])
            v_i, _ = p.getBaseVelocity(self.drone_ids[i])
            pos[i] = np.asarray(p_i, dtype=np.float32)
            vel[i] = np.asarray(v_i, dtype=np.float32)

        k_push  = float(self.sep_force)
        c_damp  = 0.5 * k_push
        f_max   = 2.0 * k_push
        m_scale = max(0.25, float(self.mass or 1.0))

        def smoothstep(x):
            x = np.clip(x, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)

        for i in range(n):
            for j in range(i + 1, n):
                dvec = pos[i] - pos[j]
                d    = float(np.linalg.norm(dvec))
                if d < 1e-8 or d >= r0:
                    continue

                if d <= r1:
                    s = 1.0
                else:
                    s = smoothstep((r0 - d) / max(1e-8, r0 - r1))

                n_ij = dvec / d

                v_rel = np.dot(vel[i] - vel[j], n_ij)

                f_mag = k_push * s - c_damp * v_rel
                f_mag = float(np.clip(f_mag, -f_max, f_max))

                Fi = (n_ij * ( f_mag * m_scale)).tolist()
                Fj = (n_ij * (-f_mag * m_scale)).tolist()
                p.applyExternalForce(self.drone_ids[i], -1, Fi, [0, 0, 0], p.WORLD_FRAME)
                p.applyExternalForce(self.drone_ids[j], -1, Fj, [0, 0, 0], p.WORLD_FRAME)

                if getattr(self, "debug_separation", False) and self.gui:
                    p.addUserDebugLine(pos[i].tolist(), pos[j].tolist(), [1, 0, 0], lifeTime=0.05)
    def _apply_obstacle_repulsion(self):
        """
        Apply soft radial repulsion forces away from static obstacles and the dynamic pursuer.

        This mirrors the pairwise separation forces but against environment geometry, so that
        drones get a gentle push away before hard collisions occur.
        """
        # nothing to do if there are no drones
        if self.num_drones <= 0:
            return

        # --- static obstacle repulsion ---
        if self.obstacle_ids:
            r_obs = float(getattr(self, "obstacle_repulse_radius", 0.7))
            k_obs = float(getattr(self, "obstacle_repulse_gain", 1.0))
            c_obs = 0.3 * k_obs
            f_obs_max = 3.0 * k_obs

            for did in self.drone_ids:
                pos_i, _ = p.getBasePositionAndOrientation(did)
                vel_i, _ = p.getBaseVelocity(did)
                pos_i = np.asarray(pos_i, dtype=np.float32)
                vel_i = np.asarray(vel_i, dtype=np.float32)

                net_force = np.zeros(3, dtype=np.float32)
                for oid in self.obstacle_ids:
                    if oid is None:
                        continue
                    opos, _ = p.getBasePositionAndOrientation(oid)
                    opos = np.asarray(opos, dtype=np.float32)

                    dvec = pos_i - opos
                    d = float(np.linalg.norm(dvec))
                    if d < 1e-6 or d > r_obs:
                        continue

                    n_hat = dvec / d
                    # distance-based scaling in [0,1]
                    s = (r_obs - d) / max(r_obs, 1e-6)
                    v_rel_n = float(np.dot(vel_i, n_hat))
                    f_mag = k_obs * s - c_obs * v_rel_n
                    f_mag = float(np.clip(f_mag, -f_obs_max, f_obs_max))
                    net_force += n_hat * f_mag

                if np.linalg.norm(net_force) > 0.0:
                    p.applyExternalForce(
                        did, -1, net_force.tolist(),
                        [0.0, 0.0, 0.0], p.WORLD_FRAME
                    )

        # --- dynamic sphere repulsion ---
        if self.dynamic_obstacle_id is not None:
            dyn_pos, _ = p.getBasePositionAndOrientation(self.dynamic_obstacle_id)
            dyn_pos = np.asarray(dyn_pos, dtype=np.float32)

            # default to danger_radius if present
            base_r = float(getattr(self, "dynamic_repulse_radius",
                                   getattr(self, "danger_radius", 0.8)))
            r_dyn = max(0.1, base_r)
            k_dyn = float(getattr(self, "dynamic_repulse_gain", 1.2))
            c_dyn = 0.3 * k_dyn
            f_dyn_max = 3.0 * k_dyn

            for did in self.drone_ids:
                pos_i, _ = p.getBasePositionAndOrientation(did)
                vel_i, _ = p.getBaseVelocity(did)
                pos_i = np.asarray(pos_i, dtype=np.float32)
                vel_i = np.asarray(vel_i, dtype=np.float32)

                dvec = pos_i - dyn_pos
                d = float(np.linalg.norm(dvec))
                if d < 1e-6 or d > r_dyn:
                    continue

                n_hat = dvec / d
                s = (r_dyn - d) / max(r_dyn, 1e-6)
                v_rel_n = float(np.dot(vel_i, n_hat))
                f_mag = k_dyn * s - c_dyn * v_rel_n
                f_mag = float(np.clip(f_mag, -f_dyn_max, f_dyn_max))
                p.applyExternalForce(
                    did, -1, (n_hat * f_mag).tolist(),
                    [0.0, 0.0, 0.0], p.WORLD_FRAME
                )


    
    def _check_collisions(self) -> bool:
        """
        Return True if a terminating collision occurred.

        Extras:
        - ignores contacts for a very short cooldown right after reset (spawn settling)
        - optional proximity early-out via getClosestPoints margin
        """
        # only skip collision checks on the very first step after reset
        if self.step_count < 1:
            return False

        prox_margin = 0.0  # can be > 0.0 to turn on "near miss" termination

        # Drone vs static obstacles
        if self.obstacle_ids:
            for i in range(self.num_drones):
                bodyA = self.drone_ids[i]

                for obs in self.obstacle_ids:
                    if obs is None:
                        continue
                    if p.getContactPoints(bodyA, obs):
                        return True
                    if prox_margin > 0.0 and p.getClosestPoints(bodyA, obs, prox_margin):
                        return True

        # Drone vs dynamic obstacle (pursuer)
        if self.dynamic_obstacle_id is not None:
            bodyB = self.dynamic_obstacle_id
            for i in range(self.num_drones):
                bodyA = self.drone_ids[i]
                if p.getContactPoints(bodyA, bodyB):
                    return True
                if prox_margin > 0.0 and p.getClosestPoints(bodyA, bodyB, prox_margin):
                    return True

        # treat "fell below z_min" as a collision
        z_min = 0.05
        for i in range(self.num_drones):
            pos, _ = p.getBasePositionAndOrientation(self.drone_ids[i])
            if pos[2] < z_min:
                return True

        # treat excessive tilt as a crash, too
        tilt_deg = float(getattr(self, "tilt_terminate_deg", 30.0))
        if tilt_deg > 0.0:
            tilt_rad = np.deg2rad(tilt_deg)
            for i in range(self.num_drones):
                _, quat = p.getBasePositionAndOrientation(self.drone_ids[i])
                roll, pitch, _ = p.getEulerFromQuaternion(quat)
                if abs(roll) > tilt_rad or abs(pitch) > tilt_rad:
                    return True

        # Drone–drone physical contact
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                if p.getContactPoints(self.drone_ids[i], self.drone_ids[j]):
                    return bool(self.terminate_on_drone_collision)



    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.num_drones, self.act_per_drone)

        # --- optional command smoothing (kills 240Hz chatter) ---
        alpha = float(getattr(self, "action_smooth_alpha", 0.0))
        if alpha > 0.0:
            action = alpha * self._prev_action + (1.0 - alpha) * action

        # --- cache action + delta-action for reward terms ---
        self.last_action_delta = action - self._prev_action
        self._prev_action = action.copy()
        self.last_action = action.copy()

        # high-level attitude + thrust control
        for i in range(self.num_drones):
            self._apply_action(i, action[i])

        # pairwise drone separation forces
        self._apply_separation_forces()
        # repulsive forces from static obstacles / dynamic pursuer
        self._apply_obstacle_repulsion()

        if not self.disable_dynamic:
            self._update_dynamic_obstacle()

        self.leader_traj_t += self.time_step
        p.stepSimulation()

        if self.gui:
            self._update_camera()

        self.collision_happened = self._check_collisions()

        obs_all = self._get_all_obs()
        global_obs = obs_all.flatten().astype(np.float32)

        rewards_all = self._compute_rewards(obs_all)  # now includes formation-break penalty too
        team_reward = float(np.mean(rewards_all))

        metrics = self.last_metrics or {}
        formation_break = bool(metrics.get("formation_break", 0.0))

        self.step_count += 1
        terminated = bool(self.collision_happened or formation_break)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "per_drone_rewards": rewards_all,
            "metrics": self.last_metrics,
        }
        return global_obs, team_reward, terminated, truncated, info



    def _spawn_drones(self, num: int):
        """
        Drones spawn here but are immediately re-placed in reset().
        Keep dynamics tame and zero initial velocity for stability.
        """
        ids: list[int] = []
        quat = p.getQuaternionFromEuler([0, 0, 0])
        for i in range(num):
            start_pos = [i * 0.3, 0.0, float(self.target_z)]
            drone_id = p.loadURDF(URDF_PATH, start_pos, quat, useFixedBase=False)
            p.changeDynamics(
                drone_id, -1,
                linearDamping=0.05, angularDamping=0.05,
                restitution=0.0,
                lateralFriction=0.8, rollingFriction=0.1, spinningFriction=0.1,
            )
            p.resetBaseVelocity(drone_id, [0, 0, 0], [0, 0, 0])
            ids.append(drone_id)
        return ids


    def _spawn_static_obstacles(self):
        """
        Fixed layout + modest random scale. Keep sizes reasonable for Stage-1 curriculum.
        """
        rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random.default_rng()

        obs_ids: list[int] = []
        scale = float(rng.uniform(0.8, 1.3))
        positions = [
            [1.0,  0.0, 0.25],
            [2.0, -1.0, 0.25],
            [2.0,  1.0, 0.25],
            [3.0,  0.0, 0.25],
        ]
        for pos in positions:
            box_id = p.loadURDF(
                os.path.join(ASSETS_DIR, "cube_small.urdf"),
                basePosition=pos, baseOrientation=[0, 0, 0, 1],
                globalScaling=scale, useFixedBase=True,
            )
            p.changeDynamics(
                box_id, -1,
                restitution=0.0,
                lateralFriction=0.9, rollingFriction=0.1, spinningFriction=0.1,
            )
            obs_ids.append(box_id)
        self.last_obstacle_scale = scale
        return obs_ids


    def _spawn_dynamic_obstacle(self):
        """
        Spawn the pursuer at drone altitude, away from drones & leader.
        Math: sample center in workspace, reject if within d_min of any drone/leader start.
        """
        rng = getattr(self, "np_random", None) or np.random.default_rng()

        XY_MIN, XY_MAX = np.array([-2.0, -2.0], np.float32), np.array([6.0, 2.0], np.float32)
        z0 = float(self.target_z)

        d_min_spawn = 0.8

        leader0 = self.leader_trajectory(0.0)

        def _sample_center():
            x0 = float(rng.uniform(0.8, 2.2))
            y0 = float(rng.uniform(-0.6, 0.6))
            return np.array([x0, y0, z0], dtype=np.float32)

        drones_pos = []
        for did in self.drone_ids:
            p_i, _ = p.getBasePositionAndOrientation(did)
            drones_pos.append(np.asarray(p_i, np.float32))
        drones_pos = np.stack(drones_pos) if len(drones_pos) else np.zeros((0, 3), np.float32)

        center = _sample_center()
        for _ in range(8):
            ok = True
            if drones_pos.size:
                if np.min(np.linalg.norm(drones_pos - center[None, :], axis=1)) < d_min_spawn:
                    ok = False
            if np.linalg.norm(center - leader0) < d_min_spawn:
                ok = False
            if ok:
                break
            center = _sample_center()

        scale = float(rng.uniform(0.7, 1.0))
        self.dynamic_scale  = scale
        self.dynamic_center = center
        self.dynamic_amp    = float(rng.uniform(0.2, 0.7))
        self.dynamic_speed  = float(rng.uniform(0.02, 0.05))
        self.dynamic_phase  = float(rng.uniform(0.0, 2.0 * np.pi))

        sphere_id = p.loadURDF(
            os.path.join(ASSETS_DIR, "sphere_small.urdf"),
            basePosition=center.tolist(), baseOrientation=[0, 0, 0, 1],
            globalScaling=scale, useFixedBase=False,
        )
        p.changeDynamics(
            sphere_id, -1,
            restitution=0.0, lateralFriction=0.5,
            linearDamping=0.05, angularDamping=0.05,
        )
        p.resetBaseVelocity(sphere_id, [0, 0, 0], [0, 0, 0])

        self.chase_target_drone = self._pick_chase_target(self.chase_mode)
        self._last_retarget_time = self._sim_time()

        return sphere_id


    def _update_dynamic_obstacle(self):
        """
        Kinematic pursuit in XY with a bounded velocity field.
        Math: v_des = v_max * \hat{pursuit} + v_lat, then first-order lag → clamp |v_xy|.
        """
        if self.dynamic_obstacle_id is None:
            return

        self.dynamic_phase += self.dynamic_speed
        y_target = self.dynamic_center[1] + self.dynamic_amp * np.sin(self.dynamic_phase)

        pos, _ = p.getBasePositionAndOrientation(self.dynamic_obstacle_id)
        vel, _ = p.getBaseVelocity(self.dynamic_obstacle_id)
        pos = np.asarray(pos, np.float32)
        vel = np.asarray(vel, np.float32)

        now_t = self._sim_time()
        tgt_id = self.chase_target_drone
        tgt_pos, _ = p.getBasePositionAndOrientation(self.drone_ids[tgt_id])
        tgt_pos = np.asarray(tgt_pos, np.float32)

        dist_xy = float(np.linalg.norm(pos[:2] - tgt_pos[:2]))
        time_up = (now_t - self._last_retarget_time) >= float(self.retarget_interval)
        reached = bool(self.retarget_on_close and (dist_xy <= float(self.retarget_close_dist)))
        if time_up or reached:
            self.chase_target_drone = self._pick_chase_target(self.chase_mode)
            self._last_retarget_time = now_t
            tgt_pos, _ = p.getBasePositionAndOrientation(self.drone_ids[self.chase_target_drone])
            tgt_pos = np.asarray(tgt_pos, np.float32)

        d_xy = tgt_pos - pos
        d_xy[2] = 0.0
        n = float(np.linalg.norm(d_xy))
        dir_xy = (d_xy / n) if n > 1e-6 else np.zeros(3, np.float32)

        aggr = float(self.dynamic_aggression)
        v_max = 0.5 + 0.7 * aggr
        k_lag = 0.25 + 0.35 * aggr

        v_des = dir_xy * v_max
        v_des[1] += 1.0 * (y_target - pos[1])
        v_des[2] = 0.0

        new_vel = vel + k_lag * (v_des - vel) * self.time_step
        s_xy = float(np.linalg.norm(new_vel[:2]))
        if s_xy > v_max:
            new_vel[:2] *= (v_max / (s_xy + 1e-8))
        new_vel[2] = 0.0

        pos_next = pos + new_vel * self.time_step
        pos_next[2] = self.target_z
        pos_next[0] = float(np.clip(pos_next[0], -2.0, 6.0))
        pos_next[1] = float(np.clip(pos_next[1], -2.0, 2.0))

        p.resetBasePositionAndOrientation(self.dynamic_obstacle_id, pos_next.tolist(), [0, 0, 0, 1])
        p.resetBaseVelocity(self.dynamic_obstacle_id, new_vel.tolist(), [0.0, 0.0, 0.0])

        if getattr(self, "debug_target", False) and self.gui:
            p.addUserDebugLine(pos_next.tolist(), tgt_pos.tolist(), [1, 0, 1], lifeTime=0.1)

    def _sim_time(self) -> float:
        """Simulation time (s) = step_count * dt."""
        return float(self.step_count) * float(self.time_step)
    
    def _pick_chase_target(self, mode: str = "random") -> int:
        """
        Choose which drone to attack. Uses the env RNG (self.np_random) for reproducibility.

        Modes:
        - "random"      : uniform over all drones, avoiding immediate repeat if possible
        - "round_robin" : 0,1,2,...,(N-1),0,...
        - "nearest"     : argmin distance in XY from sphere
        - "most_error"  : argmax formation slot error ‖pos_i - (leader+offset_i)‖
        """
        n = int(self.num_drones)
        if n <= 1:
            return 0

        rng = getattr(self, "np_random", None) or np.random.default_rng()
        mode = (mode or self.chase_mode or "random").lower()

        drone_pos = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            pos_i, _ = p.getBasePositionAndOrientation(self.drone_ids[i])
            drone_pos[i] = np.asarray(pos_i, dtype=np.float32)

        if self.dynamic_obstacle_id is not None:
            sph_pos, _ = p.getBasePositionAndOrientation(self.dynamic_obstacle_id)
            sph_pos = np.asarray(sph_pos, dtype=np.float32)
        else:
            sph_pos = self.leader_trajectory(self.leader_traj_t)

        leader_des = self.leader_trajectory(self.leader_traj_t)
        desired = np.array(
            [leader_des + self.formation_offsets[i] for i in range(n)],
            dtype=np.float32,
        )
        slot_err = np.linalg.norm(drone_pos - desired, axis=1)
        slot_err = np.nan_to_num(slot_err, nan=0.0, posinf=1e9, neginf=0.0)

        cur = int(getattr(self, "chase_target_drone", -1) or -1)

        if mode == "round_robin":
            return (cur + 1) % n

        if mode == "nearest":
            d_xy = np.linalg.norm(drone_pos[:, :2] - sph_pos[None, :2], axis=1)
            idx = int(np.argmin(d_xy))
            if idx == cur and n > 1:
                order = np.argsort(d_xy)
                for k in order:
                    if int(k) != cur:
                        return int(k)
            return idx

        if mode == "most_error":
            idx = int(np.argmax(slot_err))
            if idx == cur and n > 1:
                order = np.argsort(-slot_err)
                for k in order:
                    if int(k) != cur:
                        return int(k)
            return idx

        if n > 1:
            pool = [i for i in range(n) if i != cur] if cur in range(n) else list(range(n))
            return int(rng.choice(pool))
        return int(rng.integers(0, n))


    def _get_all_obs(self) -> np.ndarray:
        """
        Per-drone obs:
        [pos(3), vel(3), euler(3), ang_vel(3),
        neighbors((N-1)*6), static_rel(4*3), dyn_rel(3), slot_error(3)]
        Notes:
        - neighbors are ordered by ascending drone index (stable)
        - static & neighbor rel vectors are softly clipped to avoid extreme outliers
        - dyn_rel = 0 when dynamic obstacle is disabled (keeps VecNormalize sane)
        """
        all_states = [self._get_state(i) for i in range(self.num_drones)]

        static_states = []
        for oid in self.obstacle_ids:
            pos_o, _ = p.getBasePositionAndOrientation(oid)
            static_states.append(np.array(pos_o, dtype=np.float32))

        if self.dynamic_obstacle_id is None:
            dyn_pos = np.zeros(3, dtype=np.float32)
            dyn_enabled = False
        else:
            dyn_enabled = True
            dyn_pos = np.array(p.getBasePositionAndOrientation(self.dynamic_obstacle_id)[0], dtype=np.float32)

        leader_target = self.leader_trajectory(self.leader_traj_t)

        def _clip_vec(v: np.ndarray, limit: float = 5.0) -> np.ndarray:
            return np.clip(v, -limit, limit)

        obs_list = []
        for i in range(self.num_drones):
            pos_i, vel_i, euler_i, ang_i = all_states[i]

            neighbor_chunks = []
            for j in range(self.num_drones):
                if i == j:
                    continue
                pos_j, vel_j, _, _ = all_states[j]
                neighbor_chunks.append(_clip_vec(pos_j - pos_i))
                neighbor_chunks.append(_clip_vec(vel_j - vel_i))
            neighbors = (
                np.concatenate(neighbor_chunks, dtype=np.float32)
                if neighbor_chunks else np.zeros(self.neighbor_dim, dtype=np.float32)
            )

            if static_states:
                static_rel = np.concatenate([_clip_vec(s - pos_i) for s in static_states], dtype=np.float32)
            else:
                static_rel = np.zeros(self.static_obs_dim, dtype=np.float32)

            dyn_rel = (dyn_pos - pos_i) if dyn_enabled else np.zeros(3, dtype=np.float32)

            offset = self.formation_offsets.get(i, np.zeros(3, dtype=np.float32))
            desired = leader_target + offset
            slot_error = _clip_vec(desired - pos_i)

            full = np.concatenate(
                [pos_i, vel_i, euler_i, ang_i, neighbors, static_rel, dyn_rel, slot_error],
                dtype=np.float32,
            )
            obs_list.append(full)

        return np.vstack(obs_list).astype(np.float32)


    def _get_state(self, idx: int):
        pos, quat = p.getBasePositionAndOrientation(self.drone_ids[idx])
        vel_lin, vel_ang = p.getBaseVelocity(self.drone_ids[idx])

        pos = np.array(pos, dtype=np.float32)
        vel = np.array(vel_lin, dtype=np.float32)
        ang = np.array(vel_ang, dtype=np.float32)

        euler = np.array(p.getEulerFromQuaternion(quat), dtype=np.float32)
        euler = (euler + np.pi) % (2.0 * np.pi) - np.pi

        vnorm = float(np.linalg.norm(vel))
        vmax = 5.0
        if vnorm > vmax:
            vel *= (vmax / (vnorm + 1e-8))

        return pos, vel, euler, ang


    def _apply_action(self, drone_idx: int, act_vec: np.ndarray) -> None:
        # --- decode normalized actions ---
        roll_cmd, pitch_cmd, yawrate_cmd, thrust_cmd = np.asarray(act_vec, dtype=np.float32)

        roll_des = float(np.clip(roll_cmd, -1.0, 1.0)) * float(self.max_roll)         # rad
        pitch_des = float(np.clip(pitch_cmd, -1.0, 1.0)) * float(self.max_pitch)      # rad
        yaw_rate_des = float(np.clip(yawrate_cmd, -1.0, 1.0)) * float(self.max_yaw_rate)  # rad/s

        # --- state ---
        pos, vel, euler, w_world = self._get_state(drone_idx)
        roll, pitch = float(euler[0]), float(euler[1])
        z, vz = float(pos[2]), float(vel[2])

        # --- WORLD -> BODY angular velocity ---
        _, orn = p.getBasePositionAndOrientation(self.drone_ids[drone_idx])
        R = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float32).reshape(3, 3)  # body->world
        w_body = R.T @ np.asarray(w_world, dtype=np.float32)  # world->body

        # --- attitude / yaw-rate PD -> BODY torques ---
        tau_x = self.kp_roll * (roll_des - roll) - self.kd_roll * float(w_body[0])
        tau_y = self.kp_pitch * (pitch_des - pitch) - self.kd_pitch * float(w_body[1])
        tau_z = self.kp_yaw_rate * (yaw_rate_des - float(w_body[2])) - self.kd_yaw * float(w_body[2])

        tau = np.array([tau_x, tau_y, tau_z], dtype=np.float32)
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        # --- altitude PD in WORLD-Z -> desired WORLD vertical force ---
        err_z = float(self.target_z) - z
        Fz_pd = self.kp_z * err_z - self.kd_z * vz
        thrust_bias = float(np.clip(thrust_cmd, -1.0, 1.0)) * float(self.thrust_delta_scale) * float(self.hover_thrust)
        Fz_world_des = float(self.hover_thrust) + float(Fz_pd) + float(thrust_bias)

        # --- convert desired WORLD vertical force into BODY thrust (because we apply in LINK_FRAME) ---
        cos_tilt = float(np.clip(R[2, 2], 0.15, 1.0))  # body z dot world z
        thrust = float(np.clip(Fz_world_des / cos_tilt, 0.0, 3.0 * float(self.hover_thrust)))

        # --- safety ---
        if not np.isfinite(thrust):
            thrust = float(self.hover_thrust)
        if not np.all(np.isfinite(tau)):
            tau = np.zeros(3, dtype=np.float32)

        # --- apply in BODY frame ---
        p.applyExternalForce(self.drone_ids[drone_idx], -1, [0.0, 0.0, thrust], [0.0, 0.0, 0.0], p.LINK_FRAME)
        p.applyExternalTorque(self.drone_ids[drone_idx], -1, tau.tolist(), p.LINK_FRAME)
        
    def _compute_rewards(self, obs_all: np.ndarray):
        """
        Reward shaping.
        """
        leader_target = self.leader_trajectory(self.leader_traj_t)

        # obs layout per drone:
        # pos(0:3), vel(3:6), euler(6:9), ang_vel(9:12), ...
        positions  = obs_all[:, 0:3].astype(np.float32, copy=False)
        velocities = obs_all[:, 3:6].astype(np.float32, copy=False)
        eulers     = obs_all[:, 6:9].astype(np.float32, copy=False)
        ang_vels   = obs_all[:, 9:12].astype(np.float32, copy=False)

        desired_positions = np.array(
            [leader_target + self.formation_offsets.get(i, np.zeros(3, dtype=np.float32))
            for i in range(self.num_drones)],
            dtype=np.float32,
        )

        slot_vec = positions - desired_positions
        slot_err = np.linalg.norm(slot_vec, axis=1)
        h_slot   = self._huber(slot_err, float(self.huber_delta))

        mean_form_err = float(np.mean(slot_err))
        max_form_err  = float(np.max(slot_err))
        form_var      = float(np.var(slot_err))

        z_errs  = np.abs(positions[:, 2] - float(self.target_z))
        speeds  = np.linalg.norm(velocities, axis=1)
        omegas  = np.linalg.norm(ang_vels, axis=1)

        # ---- separation penalty (align with physical min_sep to avoid "unpenalized repulsion") ----
        sep_pen = 0.0
        min_pair_dist = float("inf")
        if self.num_drones > 1:
            sr_cfg = float(getattr(self, "sep_radius", 0.35))
            sr = max(sr_cfg, float(getattr(self, "min_sep", sr_cfg)))
            viol = []
            for i in range(self.num_drones):
                pi = positions[i]
                for j in range(i + 1, self.num_drones):
                    d = float(np.linalg.norm(pi - positions[j]))
                    if d < min_pair_dist:
                        min_pair_dist = d
                    if d < sr:
                        viol.append(sr - d)
            if viol:
                sep_pen = -float(self.sep_gain) * float(np.mean(viol))

        # ---- static clearance penalty (reward-level only) ----
        static_pen = 0.0
        min_static_dist = float("inf")
        if self.obstacle_ids:
            inv_d = []
            for oid in self.obstacle_ids:
                opos, _ = p.getBasePositionAndOrientation(oid)
                opos = np.array(opos, dtype=np.float32)
                d_all = np.linalg.norm(positions - opos[None, :], axis=1)
                if d_all.size:
                    d_min = float(np.min(d_all))
                    if d_min < min_static_dist:
                        min_static_dist = d_min
                d_clamped = np.clip(d_all, 0.2, 2.0)
                inv_d.append(np.mean(1.0 / d_clamped))
            static_pen = -float(self.static_clear_gain) * float(np.mean(inv_d))

        # --- dynamic obstacle shaping ---
        r_evade = 0.0
        r_safe  = 0.0
        threat  = False
        min_dyn_dist = float("inf")

        if self.dynamic_obstacle_id is not None:
            dyn_pos = np.array(p.getBasePositionAndOrientation(self.dynamic_obstacle_id)[0], dtype=np.float32)
            dists = np.linalg.norm(positions - dyn_pos[None, :], axis=1)
            min_dyn_dist = float(np.min(dists))
            threat = bool(min_dyn_dist <= float(self.threat_radius))

            centroid_pos = np.mean(positions,  axis=0)
            centroid_vel = np.mean(velocities, axis=0)

            if threat:
                away = centroid_pos - dyn_pos
                n = float(np.linalg.norm(away))
                if n > 1e-6:
                    away_dir = away / n
                    prog = float(np.dot(centroid_vel, away_dir))
                    r_evade = float(self.evade_gain) * max(0.0, prog)

            if min_dyn_dist < float(self.danger_radius):
                r_evade += -float(self.avoid_scale) * (float(self.danger_radius) - min_dyn_dist)
            if (mean_form_err < 0.5) and (min_dyn_dist > float(self.safe_radius)):
                r_safe = float(self.safe_bonus)

        # ---- core shaping terms ----
        r_form     = -float(self.form_w_mean) * float(np.mean(h_slot)) \
                    -float(self.form_w_max)  * max_form_err
        r_form_var = -float(self.form_var_gain) * form_var
        r_height   = -float(self.alt_w) * float(np.mean(z_errs))
        r_speed    = -float(self.speed_smooth_gain) * float(np.mean(speeds))

        roll_pitch = np.abs(eulers[:, :2])
        att_err = float(np.mean(roll_pitch))
        att_w = float(getattr(self, "att_w", 0.5))
        r_att = -att_w * att_err

        ang_vel_w = float(getattr(self, "ang_vel_w", 0.0))
        r_ang = -ang_vel_w * float(np.mean(omegas))

        # ---- action smoothness terms (cached in step) ----
        r_act_mag = 0.0
        r_act_rate = 0.0
        action_mag_w  = float(getattr(self, "action_mag_w", 0.0))
        action_rate_w = float(getattr(self, "action_rate_w", 0.0))

        last_action = getattr(self, "last_action", None)
        if last_action is not None and action_mag_w > 0.0:
            r_act_mag = -action_mag_w * float(np.mean(np.square(last_action)))

        last_dact = getattr(self, "last_action_delta", None)
        if last_dact is not None and action_rate_w > 0.0:
            r_act_rate = -action_rate_w * float(np.mean(np.square(last_dact)))

                # ---- workspace soft leash around leader XY ----
        wr = float(getattr(self, "workspace_radius", 2.0))
        wg = float(getattr(self, "workspace_gain", 0.4))
        d_xy = np.linalg.norm(positions[:, :2] - leader_target[None, :2], axis=1)
        excess = np.clip(d_xy - wr, 0.0, None)
        workspace_pen = -wg * float(np.mean(excess))

        # ---- terminations & per-step incentives ----
        alive_bonus = float(getattr(self, "alive_bonus", 0.0))
        r_alive = alive_bonus if not self.collision_happened else 0.0

        formation_break_mean = float(getattr(self, "formation_break_mean", 2.0))
        formation_break_max  = float(getattr(self, "formation_break_max", 3.0))
        formation_break = bool((mean_form_err > formation_break_mean) or (max_form_err > formation_break_max))

        # NOTE: collision_penalty remains here, but we will set it ~0 in the config
        collision_penalty = -float(getattr(self, "collision_penalty",
                                        getattr(self, "collision_penalty_val", 8.0))) \
                            if self.collision_happened else 0.0

        formation_break_penalty = -float(getattr(self, "formation_break_penalty", 0.0)) \
                                if (formation_break and (not self.collision_happened)) else 0.0

        # threat scaling
        if threat:
            g = float(self.form_under_threat_gain)
            r_form     *= g
            r_form_var *= g
        
                # ---- STL-style constraint margins (for blended reward) ----
        stl_margin = 0.0
        if getattr(self, "use_stl_reward", False):
            # Spec limits 
            form_limit   = float(getattr(self, "stl_form_limit", 0.8))  # max formation error
            sep_limit    = float(getattr(self, "stl_sep_limit", 0.6))   # min inter-drone distance
            obs_limit    = float(getattr(self, "stl_obs_limit", 1.0))   # min static obstacle clearance
            threat_limit = float(getattr(self, "stl_threat_limit", 1.0))# min threat distance

            # Current constraint values
            e_max = max_form_err
            d_min = min_pair_dist if np.isfinite(min_pair_dist) else float("inf")
            o_min = min_static_dist if np.isfinite(min_static_dist) else float("inf")
            t_min = min_dyn_dist if np.isfinite(min_dyn_dist) else float("inf")

            # Normalized margins: >0 = safely satisfied, 0 = on boundary, <0 = violated
            def _norm(numer, denom):
                return numer / max(denom, 1e-6)

            m_form   = _norm(form_limit   - e_max,   form_limit)
            m_sep    = _norm(d_min        - sep_limit,    sep_limit)
            m_obs    = _norm(o_min        - obs_limit,    obs_limit)
            m_threat = _norm(t_min        - threat_limit, threat_limit)

            # Conjunction: worst (most negative) margin dominates
            stl_margin = min(m_form, m_sep, m_obs, m_threat)

            stl_clip = float(getattr(self, "stl_reward_clip", 2.0))
            stl_margin = float(np.clip(stl_margin, -stl_clip, stl_clip))

        # --- combine into team reward ---
        raw_team_reward = (
            r_form + r_form_var + r_height + r_speed + r_att + r_ang +
            r_act_mag + r_act_rate +
            r_evade + r_safe + sep_pen + static_pen + workspace_pen +
            r_alive + collision_penalty + formation_break_penalty
        )

        # Add STL margin term if enabled (blended reward)
        if getattr(self, "use_stl_reward", False):
            stl_w = float(getattr(self, "stl_margin_weight", 0.0))
            raw_team_reward += stl_w * stl_margin

        reward_scale = float(getattr(self, "reward_scale", 0.01))
        team_reward = reward_scale * raw_team_reward

        rewards = [float(team_reward)] * self.num_drones


        rewards = [float(team_reward)] * self.num_drones

        self.last_metrics = {
            "mean_form_error":    mean_form_err,
            "max_form_error":     max_form_err,
            "form_var":           form_var,
            "mean_z_error":       float(np.mean(z_errs)),
            "mean_speed":         float(np.mean(speeds)),
            "mean_omega":         float(np.mean(omegas)),
            "min_dyn_distance":   min_dyn_dist,
            "min_static_distance": float(min_static_dist),
            "min_pair_distance":  float(min_pair_dist),
            "collision":          float(bool(self.collision_happened)),
            "formation_break":    float(formation_break),
            "threat":             float(threat),
            "stl_margin":         float(stl_margin) if getattr(self, "use_stl_reward", False) else 0.0,
            "r_terms": {
                "r_form":        float(r_form),
                "r_form_var":    float(r_form_var),
                "r_height":      float(r_height),
                "r_speed":       float(r_speed),
                "r_att":         float(r_att),
                "r_ang":         float(r_ang),
                "r_act_mag":     float(r_act_mag),
                "r_act_rate":    float(r_act_rate),
                "r_evade":       float(r_evade),
                "r_safe":        float(r_safe),
                "workspace_pen": float(workspace_pen),
                "sep_pen":       float(sep_pen),
                "static_pen":    float(static_pen),
                "alive":         float(r_alive),
                "collision":     float(collision_penalty),
                "formation_break": float(formation_break_penalty),
            },
        }

        return rewards




    def _update_camera(self):
        if not self.gui: return
        leader_pos, _ = p.getBasePositionAndOrientation(self.drone_ids[0])
        p.resetDebugVisualizerCamera(3.0, 35, -30, leader_pos)
        if self.debug_diamond:
            for des in self.get_desired_positions():
                top = (des + np.array([0, 0, 0.25], np.float32)).tolist()
                p.addUserDebugLine(des.tolist(), top, [0, 1, 0], lifeTime=0.1)
        if self.debug_separation and self.num_drones > 1:
            pos = [np.array(p.getBasePositionAndOrientation(iid)[0], np.float32) for iid in self.drone_ids]
            for i in range(self.num_drones):
                for j in range(i + 1, self.num_drones):
                    if np.linalg.norm(pos[i] - pos[j]) < self.min_sep:
                        p.addUserDebugLine(pos[i].tolist(), pos[j].tolist(), [1, 0, 0], lifeTime=0.1)
    def _render_frame(self) -> np.ndarray:
        """Return an RGB frame (H, W, 3) from a chase camera."""
        if not self.drone_ids:
            # nothing spawned yet
            return np.zeros((360, 640, 3), dtype=np.uint8)

        leader_pos, _ = p.getBasePositionAndOrientation(self.drone_ids[0])

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=leader_pos,
            distance=3.0,
            yaw=35.0,
            pitch=-30.0,
            roll=0.0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=16.0 / 9.0,
            nearVal=0.1,
            farVal=10.0,
        )
        width, height, rgb, _, _ = p.getCameraImage(
            width=640,
            height=360,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
        )
        frame = np.reshape(rgb, (height, width, 4))[:, :, :3]
        return frame

    def render(self, mode: str | None = None):
        """
        Gym/Gymnasium-style render.

        - "human": rely on the PyBullet GUI (already handled by `gui=True`).
        - "rgb_array": return a camera image as (H, W, 3) uint8 for video recording.
        """
        if mode is None:
            mode = self.render_mode

        if mode == "human":
            # Nothing special to do; PyBullet GUI is already showing the scene.
            return None

        if mode != "rgb_array":
            raise NotImplementedError(f"Unsupported render mode: {mode}")


        width, height = 640, 480

        # Look at the leader drone if it exists; otherwise the origin
        if self.drone_ids:
            leader_pos, _ = p.getBasePositionAndOrientation(self.drone_ids[0])
        else:
            leader_pos = [0.0, 0.0, self.target_z]

        cam_target = leader_pos
        cam_distance = 3.0
        cam_yaw = 35.0
        cam_pitch = -30.0

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target,
            distance=cam_distance,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0.0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=width / float(height),
            nearVal=0.1,
            farVal=10.0,
        )

        # pybullet returns (w, h, rgba, depth, seg)
        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
        )

        # Convert to (H, W, 3) uint8
        frame = np.reshape(rgba, (height, width, 4))[:, :, :3]
        return frame.astype(np.uint8)


    def get_desired_positions(self):
        leader_des = self.leader_trajectory(self.leader_traj_t)
        return np.array(
            [leader_des + self.formation_offsets[i] for i in range(self.num_drones)],
            dtype=np.float32,
        )

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
        super().close()
