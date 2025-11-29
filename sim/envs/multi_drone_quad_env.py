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

    - One policy controls ALL drones.
    - Action per drone: [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd] in [-1, 1].
      * roll_cmd, pitch_cmd → desired roll/pitch angles (scaled to +/- max_roll, max_pitch)
      * yaw_rate_cmd       → desired yaw rate
      * thrust_cmd         → thrust offset around hover
    - A PD attitude controller converts these into forces/torques.
    - Observation is a global vector: concatenated per-drone observations.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, num_drones: int = 5, gui: bool = False, max_steps: int = 2000):
        super().__init__()
        print("DEBUG: MultiDroneQuadEnv loaded from:", __file__)

        if num_drones < 1:
            raise ValueError("MultiDroneQuadEnv requires at least 1 drone.")

        self.num_drones = num_drones
        self.gui = gui
        if p.isConnected():
            # Reuse existing physics server (likely a GUI)
            self.physics_client = p.getConnectionInfo()['clientIndex']
        else:
            # Only open GUI if requested, otherwise DIRECT
            self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)

        p.setRealTimeSimulation(0)
        self.collision_happened = False
        self.last_metrics = {}

        # Episode length
        self.max_steps = max_steps
        self.step_count = 0

        # ------------------------------------------
        # High-level action space (per drone)
        # ------------------------------------------
        # [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd]  in [-1, 1]
        self.act_per_drone = 4
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_drones * self.act_per_drone,),
            dtype=np.float32,
        )

        # ------------------------------------------
        # Observation design (per drone)
        # ------------------------------------------
        # self_dim:
        #   3 pos (world)
        #   3 lin vel (world)
        #   3 euler angles (roll, pitch, yaw)
        #   3 angular vel (world)
        self.self_dim = 12

        # Neighbors: relative pos + vel for each other drone
        self.neighbor_dim = 6 * (self.num_drones - 1)

        # Obstacles: 4 static boxes (3D rel pos each), 1 dynamic sphere (3D rel pos)
        self.num_static_obstacles = 4
        self.static_obs_dim = 3 * self.num_static_obstacles
        self.dynamic_obs_dim = 3

        # Desired-slot error (3D) per drone
        self.leader_err_dim = 3

        # Per-drone obs size
        self.per_drone_obs_dim = (
            self.self_dim
            + self.neighbor_dim
            + self.static_obs_dim
            + self.dynamic_obs_dim
            + self.leader_err_dim
        )

        # Global observation (concatenated per-drone)
        global_obs_dim = self.per_drone_obs_dim * self.num_drones

        obs_high = np.ones(global_obs_dim, dtype=np.float32) * 1e6
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            shape=(global_obs_dim,),
            dtype=np.float32,
        )

        # ------------------------------------------
        # Physics + control
        # ------------------------------------------
        self.time_step = 1.0 / 240.0
        self.target_z = 1.0

        # Will be set after loading URDF
        self.mass = None
        self.hover_thrust = None

        # High-level limits
        self.max_roll = np.deg2rad(10.0)    # rad
        self.max_pitch = np.deg2rad(10.0)   # rad
        self.max_yaw_rate = np.deg2rad(60.0)  # rad/s

        # Inner-loop PD gains 
        self.kp_roll = 4.0
        self.kd_roll = 0.5
        self.kp_pitch = 4.0
        self.kd_pitch = 0.5
        self.kp_yaw_rate = 0.3
        self.kd_yaw = 0.05

        # Altitude (z) PD gains
        self.kp_z = 20.0   # N/m
        self.kd_z = 8.0    # N/(m/s)


        # Torque saturation (N·m) – tuned for Crazyflie-scale inertia
        self.max_torque = np.array([1e-3, 1e-3, 5e-4], dtype=np.float32)

        # Thrust limits (around hover)
        # We'll do: thrust = hover_thrust + thrust_cmd * (thrust_delta_scale * hover_thrust)
        self.thrust_delta_scale = 0.4

        # Formation layout (leader + 4 followers)
        base_offsets = [
            np.array([0.0, 0.0, 0.0]),    # leader
            np.array([0.5, -0.5, 0.0]),
            np.array([0.5,  0.5, 0.0]),
            np.array([1.0, -0.3, 0.0]),
            np.array([1.0,  0.3, 0.0]),
        ]
        if self.num_drones > len(base_offsets):
            raise ValueError(
                f"Formation offsets only defined for up to {len(base_offsets)} drones; "
                f"got num_drones={self.num_drones}."
            )
        self.formation_offsets = {i: base_offsets[i] for i in range(self.num_drones)}

        # "Time" for the leader trajectory, in SECONDS
        self.leader_traj_t = 0.0

        # Handles
        self.drone_ids = []
        self.obstacle_ids = []
        self.dynamic_obstacle_id = None
        self.dynamic_phase = 0.0

    # ------------------------------------------------------------
    # Leader trajectory (smooth, slow sine-wave)
    # ------------------------------------------------------------
    def leader_trajectory(self, t: float) -> np.ndarray:
        """
        t is in SECONDS.
        """
        x = 0.3 * t     # 0.3 m/s forward
        y = 0.4 * np.sin(0.4 * t)
        z = self.target_z
        return np.array([x, y, z], dtype=np.float32)

    # ------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Pick which drone the dynamic obstacle will chase this episode
        self.chase_target_drone = np.random.randint(self.num_drones)


        self.collision_happened = False
        self.step_count = 0

        # Only reset simulation, not reconnect physics server
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")


        # Spawn drones + obstacles
        self.drone_ids = self._spawn_drones(self.num_drones)
        self.obstacle_ids = self._spawn_static_obstacles()
        self.dynamic_obstacle_id = self._spawn_dynamic_obstacle()
        self.dynamic_phase = 0.0
        

        
        # --------------------------------------------
        # Apply randomized initial conditions
        # --------------------------------------------
        if options is not None:
            # --- Drone position jitter ---
            if "pos_jitter" in options:
                for i in range(self.num_drones):
                    base_pos, base_ori = p.getBasePositionAndOrientation(self.drone_ids[i])
                    jitter = np.array(options["pos_jitter"][i], dtype=np.float32)
                    new_pos = np.array(base_pos) + jitter
                    p.resetBasePositionAndOrientation(self.drone_ids[i], new_pos, base_ori)
        
            # --- Drone yaw jitter ---
            if "yaw_jitter" in options:
                for i in range(self.num_drones):
                    _, base_ori = p.getBasePositionAndOrientation(self.drone_ids[i])
                    yaw = float(options["yaw_jitter"][i])
                    new_ori = p.getQuaternionFromEuler([0, 0, yaw])
                    pos, _ = p.getBasePositionAndOrientation(self.drone_ids[i])
                    p.resetBasePositionAndOrientation(self.drone_ids[i], pos, new_ori)
        
            # --- Drone initial velocity jitter ---
            if "vel_jitter" in options:
                for i in range(self.num_drones):
                    vel = np.array(options["vel_jitter"][i], dtype=np.float32)
                    p.resetBaseVelocity(self.drone_ids[i], vel.tolist(), [0,0,0])
        
            # --- Static obstacle jitter ---
            if "obstacle_jitter" in options:
                jitter = np.array(options["obstacle_jitter"], dtype=np.float32)
                for oid in self.obstacle_ids:
                    pos, ori = p.getBasePositionAndOrientation(oid)
                    new_pos = np.array(pos) + jitter
                    p.resetBasePositionAndOrientation(oid, new_pos.tolist(), ori)
        
            # --- Dynamic obstacle jitter ---
            if "dynamic_jitter" in options:
                pos, ori = p.getBasePositionAndOrientation(self.dynamic_obstacle_id)
                new_pos = np.array(pos) + np.array(options["dynamic_jitter"], dtype=np.float32)
                p.resetBasePositionAndOrientation(self.dynamic_obstacle_id, new_pos.tolist(), ori)



        # Reset leader trajectory time
        self.leader_traj_t = 0.0

        # Use mass from URDF (all drones identical)
        dyn_info = p.getDynamicsInfo(self.drone_ids[0], -1)
        self.mass = dyn_info[0]
        # Initial guess: weight
        self.hover_thrust = self.mass * 9.81

        # --------------------------------------
        # AUTO-HOVER CALIBRATION
        # --------------------------------------
        # Apply constant thrust and see how the leader's altitude drifts,
        # then adjust hover_thrust so that drift is ~0.
        calib_steps = 180  # ~0.75s
        z0 = None
        z_last = None

        for s in range(calib_steps):
            for i in range(self.num_drones):
                p.applyExternalForce(
                    self.drone_ids[i],
                    -1,
                    [0.0, 0.0, self.hover_thrust],
                    [0.0, 0.0, 0.0],
                    p.WORLD_FRAME,  # purely vertical during calibration
                )
            p.stepSimulation()

            pos0, _ = p.getBasePositionAndOrientation(self.drone_ids[0])
            if s == 0:
                z0 = pos0[2]
            z_last = pos0[2]

        if z0 is not None and z_last is not None:
            drift = z_last - z0  # >0: went up, <0: went down
            T = calib_steps * self.time_step
            # From z(t) ≈ 0.5*(F/m - g)*T^2, solve for F correction
            F_correction = 2.0 * self.mass * drift / (T * T)
            self.hover_thrust -= F_correction
            # Safety clamp
            self.hover_thrust = max(self.hover_thrust, 0.01)

        # After calibration, reset drones back to clean starting pose/vel
        quat0 = p.getQuaternionFromEuler([0, 0, 0])
        for i in range(self.num_drones):
            start_pos = np.array([i * 0.3, 0.0, self.target_z], dtype=np.float32)
            p.resetBasePositionAndOrientation(self.drone_ids[i], start_pos, quat0)
            p.resetBaseVelocity(self.drone_ids[i], [0, 0, 0], [0, 0, 0])

        # --------------------------------------
        # Hover warm-up: let PD stabilize drones
        # --------------------------------------
        warmup_steps = 240  # 1 second
        zero_action = np.zeros((self.num_drones, self.act_per_drone), dtype=np.float32)
        for _ in range(warmup_steps):
            for i in range(self.num_drones):
                self._apply_action(i, zero_action[i])
            p.stepSimulation()

        # Initial observation
        obs_all = self._get_all_obs()
        global_obs = obs_all.flatten().astype(np.float32)
        return global_obs, {}

    # ------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(
            self.num_drones, self.act_per_drone
        )

        # Apply high-level actions to all drones
        for i in range(self.num_drones):
            self._apply_action(i, action[i])

        # Move dynamic obstacle
        self._update_dynamic_obstacle()

        # Advance leader trajectory using *time*
        self.leader_traj_t += self.time_step

        # Physics step
        p.stepSimulation()

        if self.gui:
            self._update_camera()

        # Observation
        obs_all = self._get_all_obs()
        global_obs = obs_all.flatten().astype(np.float32)

        # Rewards
        rewards_all = self._compute_rewards(obs_all)
        team_reward = float(np.mean(rewards_all))

        # Termination / truncation
        self.step_count += 1
        terminated = bool(self.collision_happened)
        truncated = bool(self.step_count >= self.max_steps)

        info = {"per_drone_rewards": rewards_all}
        return global_obs, team_reward, terminated, truncated, info

    # ------------------------------------------------------------
    # SPAWNING
    # ------------------------------------------------------------
    def _spawn_drones(self, num: int):
        ids = []
        for i in range(num):
            start_pos = np.array([i * 0.3, 0.0, self.target_z], dtype=np.float32)
            drone_id = p.loadURDF(
                URDF_PATH,
                start_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
            )

            p.changeDynamics(
                drone_id,
                -1,
                linearDamping=0.05,
                angularDamping=0.05,
                restitution=0.0,
                lateralFriction=0.8,
                rollingFriction=0.1,
                spinningFriction=0.1,
            )
            ids.append(drone_id)
        return ids

    def _spawn_static_obstacles(self):
        obs_ids = []
    
        # Random scale for obstacles for this episode
        scale = np.random.uniform(0.5, 2.0)
    
        positions = [
            [1.0, 0.0, 0.25],
            [2.0, -1.0, 0.25],
            [2.0,  1.0, 0.25],
            [3.0,  0.0, 0.25],
        ]
    
        for pos in positions:
            box_id = p.loadURDF(
                os.path.join(ASSETS_DIR, "cube_small.urdf"),
                basePosition=pos,
                baseOrientation=[0, 0, 0, 1],
                globalScaling=scale,      # <---- scale applied here
                useFixedBase=True,
            )
            p.changeDynamics(
                box_id,
                -1,
                restitution=0.0,
                lateralFriction=0.9,
                rollingFriction=0.1,
                spinningFriction=0.1,
            )
            obs_ids.append(box_id)
    
        # store scale for logging/MC analysis
        self.last_obstacle_scale = scale
    
        return obs_ids


    def _spawn_dynamic_obstacle(self):
        # -----------------------------
        # Episode randomization knobs
        # -----------------------------
        scale = np.random.uniform(0.5, 1.8)        # size difficulty
        x0    = np.random.uniform(1.0, 2.5)        # initial X placement
        y0    = np.random.uniform(-0.5, 0.5)       # initial Y
        z0    = np.random.uniform(0.25, 0.6)       # initial Z
        amp   = np.random.uniform(0.2, 1.0)        # motion amplitude
        speed = np.random.uniform(0.02, 0.08)      # oscillation speed
        phase = np.random.uniform(0, 2*np.pi)      # initial oscillation phase
    
        # Store parameters for motion + logging
        self.dynamic_amp = amp
        self.dynamic_speed = speed
        self.dynamic_phase = phase
        self.dynamic_scale = scale
    
        sphere_id = p.loadURDF(
            os.path.join(ASSETS_DIR, "sphere_small.urdf"),
            basePosition=[x0, y0, z0],
            baseOrientation=[0, 0, 0, 1],
            globalScaling=scale,                # <---- episode scaling applied here
            useFixedBase=False,
        )
    
        p.changeDynamics(
            sphere_id,
            -1,
            restitution=0.0,
            lateralFriction=0.5,
            linearDamping=0.1,
            angularDamping=0.1,
        )
    
        return sphere_id


    def _update_dynamic_obstacle(self):
        """
        Dynamic obstacle performs:
          - Chasing behavior toward selected drone
          - Sinusoidal oscillation in Y with randomized amplitude/speed/phase
          - Difficulty ramp as episode progresses
          - All motion uses obstacle parameters set in _spawn_dynamic_obstacle()
        """
    
        # -----------------------------------
        # 1. Get random oscillation parameters
        # -----------------------------------
        self.dynamic_phase += self.dynamic_speed
        osc_y = self.dynamic_amp * np.sin(self.dynamic_phase)
    
        # Current obstacle position
        obs_pos, _ = p.getBasePositionAndOrientation(self.dynamic_obstacle_id)
        obs_pos = np.array(obs_pos, dtype=np.float32)
    
        x0 = obs_pos[0]   # keep X from spawn
        z0 = obs_pos[2]   # keep Z from spawn
        y0 = osc_y        # sinusoidal motion added in Y
    
        # -----------------------------------
        # 2. Drone-chasing target position
        # -----------------------------------
        target_pos, _ = p.getBasePositionAndOrientation(
            self.drone_ids[self.chase_target_drone]
        )
        target_pos = np.array(target_pos, dtype=np.float32)
    
        # -----------------------------------
        # 3. Direction toward target (X/Z only)
        #    We let oscillation control Y, and chasing control X/Z.
        # -----------------------------------
        chase_dir = target_pos - obs_pos
        chase_dir[1] = 0.0    # XZ chase only, Y handled by oscillation
    
        norm = np.linalg.norm(chase_dir)
        if norm > 1e-6:
            chase_dir = chase_dir / norm
        else:
            chase_dir[:] = 0.0
    
        # -----------------------------------
        # 4. Difficulty ramp: aggression increases over time
        # -----------------------------------
        difficulty = min(1.0, self.step_count / 5000.0)
    
        max_speed = 0.3 + 0.7 * difficulty     # 0.3 → 1.0 m/s
        gain = 0.15 + 0.35 * difficulty        # PD gain ramp
    
        # -----------------------------------
        # 5. Velocity PD control toward target
        # -----------------------------------
        vel, _ = p.getBaseVelocity(self.dynamic_obstacle_id)
        vel = np.array(vel, dtype=np.float32)
    
        vel_desired = chase_dir * max_speed
    
        accel = gain * (vel_desired - vel)
        dt = self.time_step
    
        new_vel = vel + accel * dt
    
        # Clamp final speed
        speed = np.linalg.norm(new_vel)
        if speed > max_speed:
            new_vel *= max_speed / (speed + 1e-8)
    
        # -----------------------------------
        # 6. Apply motion:
        #    - New linear velocity (XZ chasing)
        #    - Oscillation-driven Y-position
        # -----------------------------------
    
        # Update velocity first
        p.resetBaseVelocity(self.dynamic_obstacle_id, new_vel.tolist(), [0, 0, 0])
    
        # Then override Y position with oscillation
        p.resetBasePositionAndOrientation(
            self.dynamic_obstacle_id,
            [obs_pos[0], y0, obs_pos[2]],
            [0, 0, 0, 1],
        )



    def _get_all_obs(self) -> np.ndarray:
        """
        Returns an array of shape (num_drones, per_drone_obs_dim).
        Per-drone obs:
            [pos (3), vel (3), euler rpy (3), ang_vel (3),
             neighbors_rel (6*(N-1)),
             static_obs_rel (3*4),
             dyn_obs_rel (3),
             desired_slot_error (3)]
        """
        all_states = [self._get_state(i) for i in range(self.num_drones)]

        static_states = [
            np.array(p.getBasePositionAndOrientation(oid)[0], dtype=np.float32)
            for oid in self.obstacle_ids
        ]

        dyn_pos = np.array(
            p.getBasePositionAndOrientation(self.dynamic_obstacle_id)[0],
            dtype=np.float32,
        )

        leader_target = self.leader_trajectory(self.leader_traj_t)

        obs_list = []

        for i in range(self.num_drones):
            pos_i, vel_i, euler_i, ang_i = all_states[i]

            # Neighbors
            neighbor_chunks = []
            for j in range(self.num_drones):
                if i == j:
                    continue
                pos_j, vel_j, _, _ = all_states[j]
                neighbor_chunks.append(
                    np.concatenate([pos_j - pos_i, vel_j - vel_i])
                )
            neighbors = (
                np.concatenate(neighbor_chunks, dtype=np.float32)
                if neighbor_chunks
                else np.zeros(self.neighbor_dim, dtype=np.float32)
            )

            static_rel = np.concatenate(
                [s - pos_i for s in static_states],
                dtype=np.float32,
            )
            dyn_rel = dyn_pos - pos_i

            desired_slot = leader_target + self.formation_offsets.get(
                i, np.zeros(3, dtype=np.float32)
            )
            desired_error = desired_slot - pos_i

            full_obs = np.concatenate(
                [
                    pos_i,
                    vel_i,
                    euler_i,
                    ang_i,
                    neighbors,
                    static_rel,
                    dyn_rel,
                    desired_error,
                ]
            )

            obs_list.append(full_obs.astype(np.float32))

        return np.vstack(obs_list)

    # ------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------
    def _get_state(self, idx: int):
        pos, quat = p.getBasePositionAndOrientation(self.drone_ids[idx])
        vel_lin, vel_ang = p.getBaseVelocity(self.drone_ids[idx])

        pos = np.array(pos, dtype=np.float32)
        vel = np.array(vel_lin, dtype=np.float32)
        ang = np.array(vel_ang, dtype=np.float32)

        euler = np.array(p.getEulerFromQuaternion(quat), dtype=np.float32)

        # Clamp linear velocity for stability in obs
        vnorm = np.linalg.norm(vel)
        vmax = 5.0
        if vnorm > vmax:
            vel = vel * (vmax / vnorm)

        return pos, vel, euler, ang

    # ------------------------------------------------------------
    # ACTION → FORCES/TORQUES (PD CONTROL)
    # ------------------------------------------------------------
    def _apply_action(self, drone_idx: int, act_vec: np.ndarray):
        """
        act_vec: [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd] in [-1, 1]
        """
        roll_cmd, pitch_cmd, yawrate_cmd, thrust_cmd = act_vec

        # Desired angles and yaw rate (clamped)
        roll_des = np.clip(roll_cmd, -1.0, 1.0) * self.max_roll
        pitch_des = np.clip(pitch_cmd, -1.0, 1.0) * self.max_pitch
        yaw_rate_des = np.clip(yawrate_cmd, -1.0, 1.0) * self.max_yaw_rate

        # Current state
        pos, vel, euler, ang = self._get_state(drone_idx)
        roll, pitch, yaw = euler
        wx, wy, wz = ang

        # ----------------------------
        # Attitude PD (roll, pitch, yaw-rate)
        # ----------------------------
        err_roll = roll_des - roll
        tau_x = self.kp_roll * err_roll - self.kd_roll * wx

        err_pitch = pitch_des - pitch
        tau_y = self.kp_pitch * err_pitch - self.kd_pitch * wy

        err_yaw_rate = yaw_rate_des - wz
        tau_z = self.kp_yaw_rate * err_yaw_rate - self.kd_yaw * wz

        tau = np.array([tau_x, tau_y, tau_z], dtype=np.float32)
        tau = np.clip(tau, -self.max_torque, self.max_torque)

        # ----------------------------
        # Altitude PD (z)
        # ----------------------------
        # Track target_z with a PD on position + vertical velocity.
        z = pos[2]
        vz = vel[2]
        err_z = self.target_z - z

        # PD force along +z (world). This is added on top of hover_thrust.
        Fz_pd = self.kp_z * err_z - self.kd_z * vz

        # ----------------------------
        # Collective thrust
        # ----------------------------
        # RL thrust_cmd is a *bias* around the altitude PD+hover term.
        thrust_bias = float(np.clip(thrust_cmd, -1.0, 1.0)) * (
            self.thrust_delta_scale * self.hover_thrust
        )

        thrust = self.hover_thrust + Fz_pd + thrust_bias

        # Clip thrust to avoid insane accelerations
        thrust = float(np.clip(thrust, 0.0, 3.0 * self.hover_thrust))

        # Apply thrust in body frame so tilt → lateral acceleration
        p.applyExternalForce(
            self.drone_ids[drone_idx],
            -1,
            [0.0, 0.0, thrust],
            [0.0, 0.0, 0.0],
            p.LINK_FRAME,
        )

        p.applyExternalTorque(
            self.drone_ids[drone_idx],
            -1,
            tau.tolist(),
            p.LINK_FRAME,
        )


    # ------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------
    def _compute_rewards(self, obs_all: np.ndarray):
        """
        obs_all: (num_drones, per_drone_obs_dim)

        Components (per drone):
            r_form      = -||pos - desired||          (clipped)
            r_height    = -|z - target_z|
            r_orient    = -(|roll| + |pitch|)
            r_angvel    = -0.05 * ||ang_vel||
            r_smooth    = -0.01 * ||vel||
            r_close     = +1.0 if formation error < 0.15 m else 0
            r_avoid     = soft penalty when obstacle is near
            r_safe      = small bonus when in formation AND obstacle far
            + collision penalty (-5) if ANY drone hits obstacle
        """
        rewards = []
        collision_penalty = 0.0

        # Collision check (once per step)
        self.collision_happened = False
        for i in range(self.num_drones):
            for obs in self.obstacle_ids + [self.dynamic_obstacle_id]:
                if p.getContactPoints(self.drone_ids[i], obs):
                    collision_penalty = -5.0
                    self.collision_happened = True
                    break
            if self.collision_happened:
                break

        leader_target = self.leader_trajectory(self.leader_traj_t)

        # Dynamic obstacle position for proximity reward
        dyn_pos = np.array(
            p.getBasePositionAndOrientation(self.dynamic_obstacle_id)[0],
            dtype=np.float32,
        )
        # Larger keep-out zones around the attacking sphere to make avoidance
        # behavior salient during training.
        danger_radius = 0.8
        safe_radius = 1.5

        for i in range(self.num_drones):
            pos = obs_all[i][0:3]
            vel = obs_all[i][3:6]
            euler = obs_all[i][6:9]
            ang = obs_all[i][9:12]

            # Desired formation position
            offset = self.formation_offsets.get(i, np.zeros(3, dtype=np.float32))
            desired = leader_target + offset

            raw_form_err = np.linalg.norm(pos - desired)

            # Penalize formation error (clipped so it doesn't dominate)
            r_form = -np.clip(raw_form_err, 0.0, 2.0)

            # Height tracking
            r_height = -abs(pos[2] - self.target_z)

            roll, pitch, _ = euler
            r_orient = -(abs(roll) + abs(pitch))

            r_angvel = -0.05 * np.linalg.norm(ang)
            r_smooth = -0.01 * np.linalg.norm(vel)

            # Bonus for being close to desired formation slot
            r_close = 1.0 if raw_form_err < 0.15 else 0.0

            # --- Obstacle avoidance ---
            d_dyn = np.linalg.norm(dyn_pos - pos)

            # Soft penalty if obstacle is within danger_radius, with a sharper
            # quadratic wall to keep drones away from the sphere. A mild penalty
            # applies inside the safe radius to preserve standoff distance.
            if d_dyn < danger_radius:
                r_avoid = -3.0 * (danger_radius - d_dyn) ** 2 - 1.0
            elif d_dyn < safe_radius:
                r_avoid = -0.5 * (safe_radius - d_dyn)
            else:
                r_avoid = 0.0

            # Small bonus when in good formation AND obstacle is far
            if (raw_form_err < 0.15) and (d_dyn > safe_radius):
                r_safe = 0.75
            else:
                r_safe = 0.0

            reward = (
                1.5 * r_form
                + 0.5 * r_height
                + 0.1 * r_orient
                + r_angvel
                + r_smooth
                + r_close
                + r_avoid
                + r_safe
                + collision_penalty
            )
            rewards.append(float(reward))
            
                # ---- aggregate metrics for logging ----
        form_errs = []
        z_errs = []
        dyn_dists = []

        leader_target = self.leader_trajectory(self.leader_traj_t)
        dyn_pos = np.array(
            p.getBasePositionAndOrientation(self.dynamic_obstacle_id)[0],
            dtype=np.float32,
        )

        for i in range(self.num_drones):
            pos = obs_all[i][0:3]
            euler = obs_all[i][6:9]
            ang = obs_all[i][9:12]

            offset = self.formation_offsets.get(i, np.zeros(3, dtype=np.float32))
            desired = leader_target + offset

            form_errs.append(np.linalg.norm(pos - desired))
            z_errs.append(abs(pos[2] - self.target_z))
            dyn_dists.append(np.linalg.norm(dyn_pos - pos))

        self.last_metrics = {
            "mean_form_error": float(np.mean(form_errs)),
            "mean_z_error": float(np.mean(z_errs)),
            "min_dyn_distance": float(np.min(dyn_dists)),
            "collision": float(self.collision_happened),
        }

        return rewards


    # ------------------------------------------------------------
    # CAMERA (GUI ONLY)
    # ------------------------------------------------------------
    def _update_camera(self):
        if not self.gui:
            return

        leader_pos, _ = p.getBasePositionAndOrientation(self.drone_ids[0])

        lx, ly, lz = leader_pos

        distance = 2.0
        yaw = 35
        pitch = -30

        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=[lx, ly, lz],
        )

    # ------------------------------------------------------------
    # Formation helpers + close
    # ------------------------------------------------------------
    def get_desired_positions(self):
        """
        Returns an array of shape (num_drones, 3).
        Each entry = leader_desired_pos + formation_offset[i]
        """
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
