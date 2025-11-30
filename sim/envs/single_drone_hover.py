import os
import time
import pybullet as p
import pybullet_data
import numpy as np


os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

if p.isConnected():
    p.disconnect()

p.connect(p.GUI)
time.sleep(1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)


p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "../assets/crazyflie/cf_assets")

p.setAdditionalSearchPath(pybullet_data.getDataPath())

drone = p.loadURDF(os.path.join(ASSETS, "cf2x.urdf"), [0, 0, 0.5])
print("Drone ID:", drone)

colCube = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
visCube = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2],
                              rgbaColor=[1, 0, 0, 1])
obstacle = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=colCube,
    baseVisualShapeIndex=visCube,
    basePosition=[0, 0, 0]
)

mass = p.getDynamicsInfo(drone, -1)[0]
hover_force = mass * 9.81
print(f"Mass = {mass:.3f} kg → Hover force = {hover_force:.3f} N")

target_height = 1.5
kp = 15.0
log_data = []
step = 0

for step in range(1000):
    pos, _ = p.getBasePositionAndOrientation(drone)
    err = target_height - pos[2]
    thrust = hover_force + kp * err
    print("Error", err)
    print("Control output (thrust):",thrust,"N")
    
    p.applyExternalForce(
        drone, -1,
        [0, 0, thrust ],
        [0, 0, 0],
        p.LINK_FRAME
    )
    
    p.stepSimulation()
    
    contacts = p.getContactPoints(drone, obstacle)
    if contacts:
        print(f"Collision! {len(contacts)} points, penetration {contacts[0][8]:.4f} m")
        print(contacts)
        p.disconnect()
    
    
    log_data.append([step / 240, pos[2], err, thrust])
    time.sleep(1/240)
    


log_data = np.array(log_data)
np.savetxt("altitude_log.csv", log_data,
           delimiter=",",
           header="time_s,height_m,error_m,thrust_N",
           comments="")

p.disconnect()

print("Simulation complete. Logged data saved to altitude_log.csv")