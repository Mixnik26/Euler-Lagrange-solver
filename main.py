import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp

class EulerLagrange():
    def __init__(self, lagrangian, h=1e-5):
        # Initialize the Euler-Lagrange equations solver
        self.lagrangian = lagrangian

        # Generate the equations of motion
        dqdot_dt = self.generate_dqdot_dt(h=h)
        self.equations_of_motion = self.generate_equations_of_motion(dqdot_dt)
    
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