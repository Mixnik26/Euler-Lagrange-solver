import main as EL
import numpy as np
import matplotlib.pyplot as plt

# Defining the lagrangian for the brachistochrone problem
g = 9.81  # gravitational acceleration
def Lagrangian(q, qdot):
    x, y = q
    if y==0:
        y=2e-5
    xdot, ydot = qdot
    return np.sqrt(np.abs((ydot**2 + xdot**2)/(-2*g*y)))

# Boundary conditions
initial_pos = np.array([0,-5])
final_pos = np.array([10,-10])

def bc(qa, qb):
    return qa[0] - initial_pos[0], qa[1] - initial_pos[1], qb[0] - final_pos[0], qb[1] - final_pos[1]

# Initialize the problem
t = np.linspace(0, 1, 50)
initial_q_array = np.array([np.linspace(0, 1, 50), np.linspace(final_pos[1],final_pos[1],50), np.zeros(50), np.zeros(50)]) # initial guess for the solution

brachistochrone = EL.EulerLagrange(Lagrangian, dimensionality=2)
brachistochrone.bvp(t, bc, initial_q_array, verbose=2)

plt.title('Brachistochrone Problem')
plt.plot(brachistochrone.q[0], brachistochrone.q[1], label='Brachistochrone Path')
plt.show()