import main as EL
import numpy as np
import matplotlib.pyplot as plt

# Defining the lagrangian for the brachistochrone problem
g = 9.81  # gravitational acceleration
def Lagrangian(q, qdot):
    y = q[0]
    ydot = qdot[0]
    return np.sqrt((ydot**2 + 1)/(2*g*y))

# Boundary conditions
initial_pos = (0,11)
final_pos = (1,10.5)

def bc(qa, qb):
    return qa[0] - initial_pos[1], qb[0] - final_pos[1]

# Initialize the problem
x = np.linspace(initial_pos[0], final_pos[0], 100)  # total displacement
initial_q_array = np.array([np.linspace(11,10.5,100), -.5*np.ones(len(x))]) # initial guess for the solution

brachistochrone = EL.EulerLagrange(Lagrangian)
brachistochrone.bvp(x, bc, initial_q_array, verbose=2)

plt.title('Brachistochrone Problem')
plt.plot(brachistochrone.t, brachistochrone.q[0], label='Brachistochrone Path')
plt.show()