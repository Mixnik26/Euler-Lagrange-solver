import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class EulerLagrange():
    def __init__(self, lagrangian, time_span: np.ndarray, initial_conditions: tuple[np.ndarray, np.ndarray], final_conditions: tuple[np.ndarray, np.ndarray] = None, initial_guess: tuple[np.ndarray, np.ndarray] = None):
        # Initialize the Euler-Lagrange equations solver
        self.lagrangian = lagrangian
        self.initial_conditions = initial_conditions
        self.time_span = time_span

        # Generate the equations of motion
        dqdot_dt = self.generate_dqdot_dt()
        equations_of_motion = self.generate_equations_of_motion(dqdot_dt)
        
        if final_conditions is None:
            # Solve the boundary value problem using solve_ivp
            self.solution = solve_ivp(equations_of_motion, (time_span[0], time_span[-1]), np.concatenate(initial_conditions), t_eval=time_span)
        else:
            # Solve the boundary value problem using solve_bvp
            initial_conditions = np.concatenate(initial_conditions)
            final_conditions = np.concatenate(final_conditions)
            initial_guess = np.concatenate(initial_guess)

            # Define the boundary conditions
            def bc(qa, qb):
                left_bc_res = np.array([qa[i] - initial_conditions[i] for i in range(len(initial_conditions)) if initial_conditions[i] is not None])
                right_bc_res = np.array([qb[i] - final_conditions[i] for i in range(len(final_conditions)) if final_conditions[i] is not None])
                return sum(left_bc_res**2), sum(right_bc_res**2)

            initial_guess = np.array([initial_conditions[i] if initial_conditions[i] is not None else initial_guess[i] for i in range(len(initial_conditions))])
            final_guess = np.array([final_conditions[i] if final_conditions[i] is not None else 0 for i in range(len(initial_conditions))])
            initial_z = np.array([np.linspace(initial_guess[i], final_guess[i], len(time_span)) for i in range(len(initial_conditions))])

            self.solution = solve_bvp(equations_of_motion, bc, time_span, initial_z, verbose=2)

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
    # Solve the brachistochrone problem using the Euler-Lagrange equations
    # Define the Lagrangian for the brachistochrone problem
    def lagrangian(t, q, qdot):
        x = q[0]
        xdot = qdot[0]
        return (x**3)/3 + (xdot**2)/2
    
    el = EulerLagrange(lagrangian, time_span=np.linspace(0, 1, 100), initial_conditions=(np.array([None]), np.array([0])), final_conditions=(np.array([1]), np.array([None])), initial_guess=(np.array([.5]), np.array([0])))

    plt.plot(el.solution.x, el.solution.y[0], label='x(t)')
    plt.show()
