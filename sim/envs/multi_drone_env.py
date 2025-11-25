import os
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import pybullet_data

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets", "crazyflie", "cf_assets")
URDF_PATH = os.path.join(ASSETS_DIR, "cf2x.urdf")


class MultiDroneEnv(gym.Env):
    """
    Centralized multi-drone formation tracking environment.

    - A single policy controls ALL drones via a vector action.
    - Observation is a single global vector containing per-drone observations.
    - Reward is a scalar "team" reward (average of per-drone rewards).
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, num_drones: int = 5, gui: bool = False):
        super().__init__()
        print("DEBUG: MultiDroneEnv loaded from:", __file__)

        if num_drones < 1:
            raise ValueError("MultiDroneEnv requires at least 1 drone.")

        self.num_drones = num_drones
        self.gui = gui
        self.physics_client = None
        self.collision_happened = False

        # Episode length
        self.max_steps = 2000
        self.step_count = 0

        # ------------------------------------------
        # Action space: 1 thrust per drone
        # ------------------------------------------
        # Single centralized policy controlling all drones.
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_drones,),  # 1D thrust per drone
            dtype=np.float32,
        )

        # ------------------------------------------
        # Observation design (per drone)
        # ------------------------------------------
        # self_dim: 3 pos + 3 vel
        self.self_dim = 6
        # neighbor_dim: for each other drone: 3 relative pos + 3 relative vel
        self.neighbor_dim = 6 * (self.num_drones - 1)
        # obstacles: 4 static boxes (3D pos each, relative), 1 dynamic sphere (3D pos, relative)
        self.num_static_obstacles = 4
        self.static_obs_dim = 3 * self.num_static_obstacles
        self.dynamic_obs_dim = 3
        # leader error term (3D): trajectory target - current leader position
        self.leader_err_dim = 3

        # Per-drone obs size
        self.per_drone_obs_dim = (
            self.self_dim
            + self.neighbor_dim
            + self.static_obs_dim
            + self.dynamic_obs_dim
            + self.leader_err_dim
        )

        # Global observation: concatenate per-drone obs for all drones
        global_obs_dim = self.per_drone_obs_dim * self.num_drones

        # Use large finite bounds instead of +/-inf
        obs_high = np.ones(global_obs_dim, dtype=np.float32) * 1e6
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            shape=(global_obs_dim,),
            dtype=np.float32,
        )

        # ------------------------------------------
        # Physics
        # ------------------------------------------
        self.time_step = 1.0 / 240.0
        self.mass = 0.032
        self.hover_force = self.mass * 9.81

        # Formation
        self.target_z = 1.0

        # Base formation layout (supports up to 5 drones)
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

        # Use the first num_drones offsets
        self.formation_offsets = {
            i: base_offsets[i] for i in range(self.num_drones)
        }

        self.leader_traj_t = 0.0  # time index for leader trajectory

        # Handles
        self.drone_ids = []
        self.obstacle_ids = []
        self.dynamic_obstacle_id = None
        self.dynamic_phase = 0.0
        

    # ------------------------------------------------------------
    # Leader trajectory (smooth sine-wave trajectory)
    # ------------------------------------------------------------
    def leader_trajectory(self, t: float) -> np.ndarray:
        # Note: t is in "steps", not seconds; coefficients tuned for that scale.
        x = 0.003 * t
        y = 0.4 * np.sin(0.01 * t)
        z = self.target_z
        return np.array([x, y, z], dtype=np.float32)

    # ------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Clear collision flag + step counter
        self.collision_happened = False
        self.step_count = 0

        # Reset physics
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Spawn drones + obstacles
        self.drone_ids = self._spawn_drones(self.num_drones)
        self.obstacle_ids = self._spawn_static_obstacles()
        self.dynamic_obstacle_id = self._spawn_dynamic_obstacle()
        self.dynamic_phase = 0.0
        
        # ---- Modify difficulty: randomize obstacle sizes ----
        scale = np.random.uniform(0.01, 50.0)  # 50% to 200%
        
        for obs_id in self.obstacle_ids:
            p.changeVisualShape(obs_id, -1, meshScale=[scale, scale, scale])
            p.changeCollisionShape(obs_id, -1, collisionFramePosition=[0,0,0], meshScale=[scale, scale, scale])

        
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





        self.leader_traj_t = 0.0

        # Global observation (flattened)
        obs_all = self._get_all_obs()  # (num_drones, per_drone_obs_dim)
        global_obs = obs_all.flatten().astype(np.float32)

        return global_obs, {}

    # ------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.num_drones)

        # Apply thrusts for all drones
        for i in range(self.num_drones):
            self._apply_action(i, action[i])

        # Move dynamic obstacle
        self._update_dynamic_obstacle()

        # Update leader's target trajectory time
        self.leader_traj_t += 1.0

        # Physics
        p.stepSimulation()

        if self.gui:
            self._update_camera()

        # Build global observation
        obs_all = self._get_all_obs()  # shape (num_drones, per_drone_obs_dim)
        global_obs = obs_all.flatten().astype(np.float32)

        # Per-drone rewards, team reward = average
        rewards_all = self._compute_rewards(obs_all)
        team_reward = float(np.mean(rewards_all))

        # Episode termination / truncation
        self.step_count += 1
        terminated = bool(self.collision_happened)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "per_drone_rewards": rewards_all,
        }

        return global_obs, team_reward, terminated, truncated, info

    # ------------------------------------------------------------
    # SPAWNING
    # ------------------------------------------------------------
    def _spawn_drones(self, num: int):
        ids = []
        for i in range(num):
            start_pos = np.array([i * 0.3, 0.0, self.target_z])

            drone_id = p.loadURDF(
                URDF_PATH,
                start_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
            )

            p.changeDynamics(
                drone_id,
                -1,
                linearDamping=0.2,
                angularDamping=0.2,
                restitution=0.0,
                lateralFriction=0.8,
                rollingFriction=0.1,
                spinningFriction=0.1,
            )

            ids.append(drone_id)

        return ids

    def _spawn_static_obstacles(self):
        obs_ids = []
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
                useFixedBase=True,
            )
            p.changeDynamics(
                box_id,
                -1,
                restitution=0.0,      # no bounce
                lateralFriction=0.9,
                rollingFriction=0.1,
                spinningFriction=0.1,
            )
            obs_ids.append(box_id)

        return obs_ids

    def _spawn_dynamic_obstacle(self):
        sphere_id = p.loadURDF(
            os.path.join(ASSETS_DIR, "sphere_small.urdf"),
            basePosition=[1.5, 0.0, 0.3],
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
        self.dynamic_phase += 0.03
        y = 0.6 * np.sin(self.dynamic_phase)
        p.resetBasePositionAndOrientation(
            self.dynamic_obstacle_id,
            [1.5, y, 0.3],
            [0, 0, 0, 1],
        )

    # ------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------
    def _get_all_obs(self) -> np.ndarray:
        """
        Returns an array of shape (num_drones, per_drone_obs_dim).
        Each row is that drone's local observation:
            [pos, vel, neighbors_rel, static_obs_rel, dyn_obs_rel, leader_error]
        """
        # Get states for all drones once
        all_states = [self._get_state(i) for i in range(self.num_drones)]

        # Static obstacles (world positions)
        static_states = [
            np.array(p.getBasePositionAndOrientation(oid)[0], dtype=np.float32)
            for oid in self.obstacle_ids
        ]

        # Dynamic obstacle (world position)
        dyn_pos = np.array(
            p.getBasePositionAndOrientation(self.dynamic_obstacle_id)[0],
            dtype=np.float32,
        )

        # Leader trajectory & current leader pos
        leader_target = self.leader_trajectory(self.leader_traj_t)
        leader_pos, _ = all_states[0]

        # Leader error: target - current position (same for all drones)
        leader_error = leader_target - leader_pos

        obs_list = []

        for i in range(self.num_drones):
            pos_i, vel_i = all_states[i]

            # Neighbors: relative pos/vel of all other drones
            neighbor_chunks = []
            for j in range(self.num_drones):
                if i == j:
                    continue
                pos_j, vel_j = all_states[j]
                neighbor_chunks.append(
                    np.concatenate([pos_j - pos_i, vel_j - vel_i])
                )

            if len(neighbor_chunks) > 0:
                neighbors = np.concatenate(neighbor_chunks, dtype=np.float32)
            else:
                # num_drones == 1 -> no neighbors
                neighbors = np.zeros(self.neighbor_dim, dtype=np.float32)

            # Static obstacles: relative positions
            static_rel = np.concatenate(
                [s - pos_i for s in static_states],
                dtype=np.float32,
            )

            # Dynamic obstacle: relative position
            dyn_rel = dyn_pos - pos_i

            full_obs = np.concatenate(
                [
                    pos_i,
                    vel_i,
                    neighbors,
                    static_rel,
                    dyn_rel,
                    leader_error,
                ]
            )

            obs_list.append(full_obs.astype(np.float32))

        # Shape: (num_drones, per_drone_obs_dim)
        return np.vstack(obs_list)

    # ------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------
    def _get_state(self, idx: int):
        pos, _ = p.getBasePositionAndOrientation(self.drone_ids[idx])
        vel_lin, _ = p.getBaseVelocity(self.drone_ids[idx])

        pos = np.array(pos, dtype=np.float32)
        vel = np.array(vel_lin, dtype=np.float32)

        # Clamp velocity
        vnorm = np.linalg.norm(vel)
        vmax = 5.0  # m/s cap
        if vnorm > vmax:
            vel = vel * (vmax / vnorm)

        return pos, vel

    # ------------------------------------------------------------
    # ACTION
    # ------------------------------------------------------------
    def _apply_action(self, drone_idx: int, thrust_ratio: float):
        thrust_ratio = float(np.clip(thrust_ratio, 0.0, 1.0))
        # 0.5 -> hover, 1.0 -> ~2g upward, 0.0 -> free fall
        thrust = thrust_ratio * 2.0 * self.hover_force
        p.applyExternalForce(
            self.drone_ids[drone_idx],
            -1,
            [0.0, 0.0, thrust],
            [0.0, 0.0, 0.0],
            p.LINK_FRAME,
        )

    # ------------------------------------------------------------
    # REWARD
    # ------------------------------------------------------------
    def _compute_rewards(self, obs_all: np.ndarray):
        """
        obs_all: shape (num_drones, per_drone_obs_dim)

        Per-drone reward:
            r_form   = -||pos - desired||
            r_hover  = -|z - target_z|
            r_smooth = -0.05 * ||vel||
            + collision penalty (-8) if any drone hits any obstacle
        """
        rewards = []
        collision_penalty = 0.0

        # Collision check with obstacles (shared penalty)
        for i in range(self.num_drones):
            for obs in self.obstacle_ids + [self.dynamic_obstacle_id]:
                if p.getContactPoints(self.drone_ids[i], obs):
                    collision_penalty = -8.0
                    self.collision_happened = True
                    break
            if self.collision_happened:
                break

        # Leader target (for this step)
        leader_target = self.leader_trajectory(self.leader_traj_t)

        for i in range(self.num_drones):
            # First 3: pos, next 3: vel (by construction in _get_all_obs)
            pos = obs_all[i][0:3]
            vel = obs_all[i][3:6]

            # Desired formation position for drone i
            offset = self.formation_offsets.get(i, np.zeros(3, dtype=np.float32))
            desired = leader_target + offset

            # Formation tracking error
            form_err = np.linalg.norm(pos - desired)
            r_form = -form_err

            # Height tracking
            r_hover = -abs(pos[2] - self.target_z)

            # Smooth motion (penalize speed)
            r_smooth = -0.05 * np.linalg.norm(vel)

            reward = r_form + r_hover + r_smooth + collision_penalty
            rewards.append(float(reward))

        return rewards

    # ------------------------------------------------------------
    # CAMERA FOLLOW MODE (GUI ONLY)
    # ------------------------------------------------------------
    def _update_camera(self):
        if not self.gui:
            return

        # Leader position
        leader_pos, _ = p.getBasePositionAndOrientation(self.drone_ids[0])
        lx, ly, lz = leader_pos

        # Camera parameters
        distance = 2.8
        yaw = 35
        pitch = -25

        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=[lx, ly, lz],
        )

    # ------------------------------------------------------------
    # CLOSE
    # ------------------------------------------------------------
    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
        super().close()
