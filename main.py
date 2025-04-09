import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp

class EulerLagrange():
    def __init__(self, lagrangian, explicit_t_dependence: bool = False, ignorable_coordinates: np.ndarray[bool] = None, h: float = 1e-5):
        # Initialize the Euler-Lagrange equations solver
        self.ignorable_coordinates = ignorable_coordinates
        # Check if the Lagrangian has a t argument to determine if it has explicit t dependence
        explicit_t_dependence = 't' in lagrangian.__code__.co_varnames[:lagrangian.__code__.co_argcount]
        self.explicit_t_dependence = explicit_t_dependence
        # If theres no explicit t dependence, we can generate the Hamiltonian of the system
        if not explicit_t_dependence:
            ex_lagrangian = lagrangian
            def lagrangian(t, q, qdot):
                return ex_lagrangian(q, qdot)
            
            self.lagrangian = lagrangian
            self.generate_hamiltonian(h)
        else: # Otherwise, we just use the Lagrangian
            self.lagrangian = lagrangian
            self.hamiltonian = None

        # Generate the equations of motion
        dqdot_dt = self.generate_dqdot_dt(h=h)
        self.equations_of_motion = self.generate_equations_of_motion(dqdot_dt)
    
    def generate_equations_of_motion(self, dqdot_dt):
        # Generate the equations of motion from the Lagrangian as a first order system
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

    def generate_hamiltonian(self, h=1e-5):
        if self.explicit_t_dependence:
            print("Warning! Hamiltonian is only conserved if the Lagrangian has no explicit t dependence.")
        
        # Define the generalized momenta
        dL_dqdot = lambda t, q, qdot: (self.lagrangian(t, q, qdot + h) - self.lagrangian(t, q, qdot - h)) / (2 * h)
        self.p = dL_dqdot

        # Define the Hamiltonian
        def hamiltonian(t, q, qdot):
            return sum(dL_dqdot(t, q, qdot) * qdot - self.lagrangian(t, q, qdot))
        self.hamiltonian = hamiltonian

        return hamiltonian
    
    def ivp(self, time_span: np.ndarray, initial_conditions: tuple[np.ndarray, np.ndarray], t_eval=None, *args):
        # Solve the initial value problem using solve_ivp
        if t_eval is None:
            t_eval = time_span

        self.solution = solve_ivp(self.equations_of_motion, (time_span[0], time_span[-1]), np.concatenate(initial_conditions), t_eval=t_eval, *args)
        self.q = self.solution.y[:len(initial_conditions[0])]
        self.qdot = self.solution.y[len(initial_conditions[0]):]
        self.t = self.solution.t

        # Check if the Hamiltonian is conserved
        if not self.explicit_t_dependence:
            self.H_arr = self.hamiltonian(self.t, self.q, self.qdot)
            self.H = self.H_arr[0]
            self.H_final = self.H_arr[-1]
            self.H_diff = self.H_final - self.H
            if self.H_diff/self.H > 5e-2:
                print(f"Warning! Hamiltonian numerically varied by more that 5%: delta H = {(100*self.H_diff/self.H):.1f}%")

        return self.solution