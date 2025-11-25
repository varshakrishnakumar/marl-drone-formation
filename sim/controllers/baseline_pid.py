import numpy as np


class FormationPIDController:
    def __init__(self, env, kp_xy=1.0, ki_xy=0.0, kd_xy=0.5, kp_yaw=0.5, kd_yaw=0.1):
        """
        Outer-loop PID controller that maps per-drone position errors
        to [roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd].
        """
        self.env = env
        self.num_drones = env.num_drones
        self.per_drone_dim = env.per_drone_obs_dim
        self.dt = env.time_step

        # Position PID gains (same for x and y)
        self.kp_xy = kp_xy
        self.ki_xy = ki_xy
        self.kd_xy = kd_xy

        # Yaw PD gains (we'll try to hold yaw ≈ 0)
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw

        # Integral state for x,y per drone: shape (num_drones, 2)
        self.int_xy = np.zeros((self.num_drones, 2), dtype=np.float32)

    def reset(self):
        self.int_xy[:] = 0.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: global observation, shape (global_obs_dim,)
        returns: action, shape (num_drones * 4,)
        """
        # Reshape into per-drone chunks
        obs = obs.reshape(self.num_drones, self.per_drone_dim)

        # Current positions and velocities per drone
        # obs layout: [pos(3), vel(3), euler(3), ang_vel(3), ...]
        pos = obs[:, 0:3]          # (N, 3)
        vel = obs[:, 3:6]          # (N, 3)
        euler = obs[:, 6:9]        # (N, 3)
        ang_vel = obs[:, 9:12]     # (N, 3)

        # Desired positions from env's formation helper
        desired_pos = self.env.get_desired_positions()  # (N, 3)

        # Position error in x,y
        err_pos_xy = desired_pos[:, 0:2] - pos[:, 0:2]      # (N, 2)
        vel_xy = vel[:, 0:2]                                # (N, 2)

        # Update integrals
        self.int_xy += err_pos_xy * self.dt

        # PID for x,y ⇒ "desired accelerations" in x,y
        acc_xy = (
            self.kp_xy * err_pos_xy
            + self.ki_xy * self.int_xy
            - self.kd_xy * vel_xy
        )  # (N, 2)

        # ------------------------------------------------------------------
        # Map desired horizontal accelerations → roll/pitch commands
        #
        # Very simple linearized mapping:
        #   pitch_cmd ∝ +acc_x  (tilt forward/back)
        #   roll_cmd  ∝ -acc_y  (tilt left/right)
        #
        # Then we normalize to [-1, 1] by some scale (tunable).
        # ------------------------------------------------------------------
        acc_scale = 2.0  # tune this so that commands typically live in [-1, 1]

        pitch_cmd = np.clip(acc_xy[:, 0] * acc_scale, -1.0, 1.0)
        roll_cmd  = np.clip(-acc_xy[:, 1] * acc_scale, -1.0, 1.0)

        # ------------------------------------------------------------------
        # Yaw control: hold yaw ≈ 0 using PD on yaw
        # ------------------------------------------------------------------
        yaw = euler[:, 2]
        wz = ang_vel[:, 2]

        yaw_err = -yaw  # target yaw = 0
        yaw_rate_des = self.kp_yaw * yaw_err - self.kd_yaw * wz

        # Convert desired yaw rate to command in [-1, 1]
        yaw_rate_cmd = yaw_rate_des / self.env.max_yaw_rate
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -1.0, 1.0)

        # ------------------------------------------------------------------
        # Altitude: use inner-loop PD; we just bias thrust around hover.
        # For a simple baseline, keep thrust_cmd = 0 (i.e., hover).
        # ------------------------------------------------------------------
        thrust_cmd = np.zeros(self.num_drones, dtype=np.float32)

        # Stack into (N, 4) then flatten
        actions = np.stack([roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd], axis=1)
        return actions.astype(np.float32).flatten()
