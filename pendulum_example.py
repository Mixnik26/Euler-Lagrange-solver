import main as EL
import numpy as np
import matplotlib.pyplot as plt

g = 9.81  # gravitational acceleration
l = 1.0   # length of the pendulum
m = 1.0   # mass of the pendulum bob

def Lagrangian(q, qdot):
    # Define the Lagrangian for a simple pendulum
    theta = q[0]
    thetadot = qdot[0]
    return (1/2)*m*(l*thetadot)**2 - m*g*l*(1 - np.cos(theta))

t = np.linspace(0, 10, 100)  # time span
q0 = np.array([np.pi/4])  # initial angle (45 degrees)
qdot0 = np.array([0])  # initial angular velocity

pendulum = EL.EulerLagrange(Lagrangian)
pendulum.ivp(t, (q0, qdot0))

plt.title('Pendulum Motion')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.plot(pendulum.t, pendulum.q[0])
plt.show()