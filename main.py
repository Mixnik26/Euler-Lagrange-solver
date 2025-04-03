import numpy as np
import scipy.optimize as op
from scipy.integrate import solve_bvp

class EulerLagrange():
    def __init__(self, lagrangian, initial_conditions: tuple[np.ndarray, np.ndarray], time_span: tuple):
        # Initialize the Euler-Lagrange equations solver
        self.lagrangian = lagrangian
        self.initial_conditions = initial_conditions
        self.time_span = time_span

        # Generate the equations of motion
        equations_of_motion = self.generate_equations_of_motion()

    def generate_equations_of_motion(self):
        # Generate the equations of motion from the Lagrangian
        def equations_of_motion(t, y):
            q = y[:len(y)//2]
            p = y[len(y)//2:]
            dqdt = p
            dpdt = -self.lagrangian(q, p).dot(np.gradient(q))
            return np.concatenate((dqdt, dpdt))

        return equations_of_motion
