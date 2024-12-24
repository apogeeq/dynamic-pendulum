import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
dt = 1 / 240  # pybullet simulation step
x1 = 1  # starting position (radian)
x2 = 0.0  # Initial velocity (rad/s)
simulation_time = 5  # seconds
g = 10  # gravity

# Connect to PyBullet
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("./simple.urdf", useFixedBase=True)

# Simulation parameters 2
link_state = p.getLinkState(boxId, 1)  # Link status link_1
link_com_pos = link_state[0]  # Center of mass link_1
eef_state = p.getLinkState(boxId, 2)  # Link state link_eef
eef_com_pos = eef_state[0]  # Center of mass link_eef
l = np.linalg.norm(np.array(eef_com_pos) - np.array(link_com_pos))  # length
m = p.getDynamicsInfo(boxId, 2)[0]  # mass
d = -p.getDynamicsInfo(boxId, 2)[8]  # angular_damping

# Remove damping forces
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# Move to the starting position
p.setJointMotorControl2(
    bodyIndex=boxId, jointIndex=1, targetPosition=x1,
      controlMode=p.POSITION_CONTROL
)
for _ in range(1000):
    p.stepSimulation()

# Turn off the motor for free motion
p.setJointMotorControl2(
    bodyIndex=boxId, jointIndex=1, targetVelocity=0,
      controlMode=p.VELOCITY_CONTROL, force=0
)

# Data collection
time_list = []
angle_list = []
steps = int(simulation_time / dt)

for step in range(steps):
    # Get the joint angle and velocity
    joint_state = p.getJointState(boxId, 1)
    joint_angle = joint_state[0]  # Position of the joint
    joint_velocity = joint_state[1]  # Velocity of the joint

    # Collect data
    angle_list.append(joint_angle)
    time_list.append(step * dt)

    # Step the simulation
    p.stepSimulation()
    time.sleep(dt)

# Disconnect from PyBullet
p.disconnect()

# Time array
num_steps = int(simulation_time / dt)
time = np.linspace(0, simulation_time, num_steps)

# Arrays to store results
x1_array = np.zeros(num_steps)
x2_array = np.zeros(num_steps)

# Simulation loop (analytical)
for i in range(num_steps):
    t = i * dt

    # Calculate acceleration
    x2_dot = -(g / l) * np.sin(x1) - (d / (m * l ** 2)) * x2

    # Integrate using Euler's method
    x2 += x2_dot * dt
    x1 += x2 * dt

    # Store current state
    x1_array[i] = x1
    x2_array[i] = x2

# Calculate metrics
angle_array = np.array(angle_list)
time_array = np.array(time_list)

# L_2
L_2 = np.sum((angle_array - x1_array) ** 2)

# L_inf
L_inf = np.max(np.abs(angle_array - x1_array))

# Plot the angle vs time
plt.figure(figsize=(10, 5))
plt.plot(time_list, angle_list, label="Joint Angle (rad)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title(f"Joint Angle Over Time Pybullet")
plt.legend()
plt.grid()
plt.show()

# Plot the angle vs time (analytical)
plt.figure(figsize=(10, 5))
plt.plot(time, x1_array, label="Position x(t)", linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.title(f"Pendulum Motion")
plt.grid()
plt.legend()
plt.show()

print(f"L_2: {L_2:.4f}")
print(f"L_inf: {L_inf:.4f}")
