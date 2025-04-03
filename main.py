from operator import eq
import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp

class EulerLagrange():
    def __init__(self, lagrangian, time_span: tuple, initial_conditions: tuple[np.ndarray, np.ndarray], final_conditions: tuple[np.ndarray, np.ndarray] = None):
        # Initialize the Euler-Lagrange equations solver
        self.lagrangian = lagrangian
        self.initial_conditions = initial_conditions
        self.time_span = time_span

        # Generate the equations of motion
        dqdot_dt = self.generate_dqdot_dt()
        equations_of_motion = self.generate_equations_of_motion(dqdot_dt)

        
        if final_conditions is None:
            # Solve the boundary value problem using solve_ivp
            self.solution = solve_ivp(equations_of_motion, time_span, np.concatenate(initial_conditions), t_eval=np.linspace(time_span[0], time_span[1], 100))
        else:
            # Solve the boundary value problem using solve_bvp
            self.solution = solve_bvp(equations_of_motion, lambda t: np.concatenate(initial_conditions), time_span, np.concatenate(initial_conditions), np.concatenate(final_conditions))

    def generate_equations_of_motion(self, dqdot_dt):
        # Generate the equations of motion from the Lagrangian
        def equations_of_motion(t, z):
            q = z[:len(z)//2]
            qdot = z[len(z)//2:]
            dqdt = qdot
            dqdotdt = dqdot_dt(t, q, qdot)
            return np.concatenate((dqdt, dqdotdt))
        
        return equations_of_motion
    
    def generate_dqdot_dt(self, h=1e-5):
        # Generate the time derivatives of the generalized coordinates using the Euler-Lagrange equations and finite difference methods
        L = self.lagrangian
        dL_dq = lambda t, q, qdot: (L(t, q + h, qdot) - L(t, q - h, qdot)) / (2 * h)
        d2L_dtdqdot = lambda t, q, qdot: (L(t + h, q, qdot + h) - L(t + h, q, qdot - h) - L(t - h, q, qdot + h) + L(t - h, q, qdot - h)) / (4 * h**2)
        d2L_dqdqdot = lambda t, q, qdot: (L(t, q + h, qdot + h) - L(t, q + h, qdot - h) - L(t, q - h, qdot + h) + L(t, q - h, qdot - h)) / (4 * h**2)
        d2L_dqdot2 = lambda t, q, qdot: (L(t, q, qdot + h) - 2 * L(t, q, qdot) + L(t, q, qdot - h)) / (h**2)
        self.d2L_dqdot2 = d2L_dqdot2
        dqdot_dt = lambda t, q, qdot: (dL_dq(t, q, qdot) - d2L_dtdqdot(t, q, qdot) - qdot * d2L_dqdqdot(t, q, qdot)) / d2L_dqdot2(t, q, qdot)
        return dqdot_dt

if __name__ == "__main__":
    # Example Lagrangian: L = 1/2 m v^2 - V(x)
    def lagrangian(t, q, qdot):
        m = 1.0  # mass
        k = 1.0  # spring constant
        x = q[0]
        v = qdot[0]
        return 0.5 * m * v**2 - 0.5 * k * x**2

    # Initial conditions: [x(0), v(0)]
    initial_conditions = (np.array([1.0]), np.array([0.0]))

    # Time span for the simulation
    time_span = (0, 10)

    # Create an instance of the EulerLagrange class and solve the equations of motion
    el_solver = EulerLagrange(lagrangian, time_span, initial_conditions)

    # Print the solution
    print(el_solver.solution.y)