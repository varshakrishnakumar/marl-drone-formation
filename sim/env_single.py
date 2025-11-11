import os
import time
import pybullet as p
import pybullet_data
import numpy as np

# --- Fix Windows text cutoff / DPI scaling issue ---
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

# --- Start a fresh GUI session (avoid leftover camera states) ---
if p.isConnected():
    p.disconnect()

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)

# --- Reset camera to a consistent default view each run ---
p.resetDebugVisualizerCamera(
    cameraDistance=3.0,
    cameraYaw=50.0,
    cameraPitch=-30.0,
    cameraTargetPosition=[0, 0, 0]
)

# Enable basic controls
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)

# --- Load world elements ---
plane = p.loadURDF("plane.urdf")

drone = p.loadURDF(
    r"C:\Users\ronak\anaconda3\envs\pybullet_env\bullet3-master\data\Quadrotor\quadrotor.urdf",
    [0, 0, 1]
)
print("Drone ID:", drone)

# Obstacle (static red cube)
colCube = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
visCube = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2],
                              rgbaColor=[1, 0, 0, 1])
obstacle = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=colCube,
    baseVisualShapeIndex=visCube,
    basePosition=[1, 0, 0.2]
)

# --- Hover control setup ---
mass = p.getDynamicsInfo(drone, -1)[0]
hover_force = mass * 9.81
print(f"Mass = {mass:.3f} kg → Hover force = {hover_force:.3f} N")

target_height = 1.5
kp = 15.0  # proportional gain
log_data = []
step = 0

# --- Simulation loop of 20 seconds---
for step in range(1000):
    pos, _ = p.getBasePositionAndOrientation(drone)
    err = target_height - pos[2]
    thrust = hover_force + kp * err
    print("Error", err)
    print("Control output (thrust):",thrust,"N")
    # Apply force in body frame (vertical thrust)
    p.applyExternalForce(
        drone, -1,
        [0, 0, thrust],
        [0, 0, 0],
        p.LINK_FRAME
    )
    p.stepSimulation()
    # Log: time (s), height (m), error (m), thrust (N)
    log_data.append([step / 240, pos[2], err, thrust])
    time.sleep(1/240) #time step


# --- Save data to file ---
log_data = np.array(log_data)
np.savetxt("altitude_log.csv", log_data,
           delimiter=",",
           header="time_s,height_m,error_m,thrust_N",
           comments="")

# --- Cleanly close connection after run ---
p.disconnect()

print("Simulation complete. Logged data saved to altitude_log.csv")