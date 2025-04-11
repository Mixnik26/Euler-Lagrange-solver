import main as EL
import numpy as np
import matplotlib.pyplot as plt

# Defining the lagrangian for the brachistochrone problem
g = 9.81  # gravitational acceleration
def Lagrangian(q, qdot):
    x, y = q
    xdot, ydot = qdot
    return np.sqrt((ydot**2 + xdot**2)/(2*g*y))

# Boundary conditions
initial_pos = np.array([0,11])
final_pos = np.array([1,10.5])

def bc(qa, qb):
    return qa[0] - initial_pos[0], qa[1] - initial_pos[1], qb[0] - final_pos[0], qb[1] - final_pos[1]

# Initialize the problem
t = np.linspace(0, 1, 100)
initial_q_array = np.array([np.linspace(0, 1, 100), np.linspace(10.5,10.5,100), np.zeros(100), np.zeros(100)]) # initial guess for the solution

brachistochrone = EL.EulerLagrange(Lagrangian)
brachistochrone.bvp(t, bc, initial_q_array, verbose=2)

plt.title('Brachistochrone Problem')
plt.plot(brachistochrone.q[0], brachistochrone.q[1], label='Brachistochrone Path')
plt.show()