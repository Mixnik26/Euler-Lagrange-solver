import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp

class EulerLagrange():
    '''
    Class to handle the Euler-Lagrange equations of motion for a system defined by a Lagrangian.
    The class can handle both explicit and implicit time dependence of the Lagrangian.
    '''
    def __init__(self, lagrangian, dimensionality: int, explicit_t_dependence: bool = False, ignorable_coordinates: np.ndarray[bool] = None, h: float = 1e-5):
        # Initialize the Euler-Lagrange equations solver
        self.ignorable_coordinates = ignorable_coordinates
        self.dimensionality = dimensionality
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
        '''
        Uses dqdot_dt to generate the equations of motion from the Lagrangian.
        '''
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
        d = self.dimensionality

        def h_arr(i):
            arr = np.zeros(d)
            arr[i] = h
            return arr
        
        dL_dq = lambda t, q, qdot: np.array([
            (L(t, q + h_arr(i), qdot) - L(t, q - h_arr(i), qdot)) / (2 * h) for i in range(d)])
        
        d2L_dtdqdot = lambda t, q, qdot: np.array([
            (L(t + h_arr(i), q, qdot + h_arr(i)) - L(t + h_arr(i), q, qdot - h_arr(i)) - L(t - h_arr(i), q, qdot + h_arr(i)) + L(t - h_arr(i), q, qdot - h_arr(i))) / (4 * h**2) for i in range(d)])
        
        d2L_dqdqdot = lambda t, q, qdot: np.array([
            (L(t, q + h_arr(i), qdot + h_arr(i)) - L(t, q + h_arr(i), qdot - h_arr(i)) - L(t, q - h_arr(i), qdot + h_arr(i)) + L(t, q - h_arr(i), qdot - h_arr(i))) / (4 * h**2) for i in range(d)])
        
        d2L_dqdot2 = lambda t, q, qdot: np.array([
            (L(t, q, qdot + h_arr(i)) - 2 * L(t, q, qdot) + L(t, q, qdot - h_arr(i))) / (h**2) for i in range(d)])

        self.not_warned_d2L_dqdot2 = True
        def dqdot_dt(t, q, qdot):
            try:
                return (dL_dq(t, q, qdot) - d2L_dtdqdot(t, q, qdot) - qdot * d2L_dqdqdot(t, q, qdot)) / d2L_dqdot2(t, q, qdot)
            except ZeroDivisionError:
                if self.not_warned_d2L_dqdot2:
                    print("Warning! d2L/dqdot2 is zero. The equations of motion are not well-defined.")
                    self.not_warned_d2L_dqdot2 = False
                return (dL_dq(t, q, qdot) - d2L_dtdqdot(t, q, qdot) - qdot * d2L_dqdqdot(t, q, qdot)) / d2L_dqdot2(t, q, qdot)
        return dqdot_dt

    def generate_hamiltonian(self, h=1e-5):
        '''
        Generates a function that computes the Hamiltonian of the system from the Lagrangian for a given t, q, and qdot
        '''
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
    
    def ivp(self, time_span: np.ndarray, initial_conditions: tuple[np.ndarray, np.ndarray], t_eval=None, **kwargs):
        '''
        Solves the initial value problem using solve_ivp given by the Lagrangian using solve_ivp.
        Inputs:
        - time_span: array of time points to evaluate the solution at
        - initial_conditions: tuple of initial conditions (q0, qdot0)
        - t_eval: array of time points to evaluate the solution at (optional)
        - **kwargs: additional arguments to pass to solve_ivp
        Outputs:
        - solution: the solution object returned by solve_ivp
        '''
        # Solve the initial value problem using solve_ivp
        if t_eval is None:
            t_eval = time_span

        self.solution = solve_ivp(self.equations_of_motion, (time_span[0], time_span[-1]), np.concatenate(initial_conditions), t_eval=t_eval, **kwargs)
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
    
    def bvp(self, time_span: np.ndarray, bc, initial_q_array: np.ndarray, **kwargs):
        '''
        Solves the boundary value problem using solve_bvp given by the Lagrangian using solve_bvp.
        Inputs:
        - time_span: array of time points to evaluate the solution at
        - bc: function that defines the boundary conditions
        - initial_q_array: array of initial conditions to be passed to solve_bvp. Must be of shape (2*d, n) where d is the dimensionality of the system and n is the number of time points.
        - **kwargs: additional arguments to pass to solve_bvp
        Outputs:
        - solution: the solution object returned by solve_bvp
        '''
        def fun(T, z):
            dz_dt = []
            for t,q in zip(T,z.T):
                dz_dt.append(self.equations_of_motion(t, q))
            return np.array(dz_dt).T


        self.solution = solve_bvp(fun, bc, time_span, initial_q_array, **kwargs)

        if self.solution.success:
            print("Boundary value problem solved successfully.")
        else:
            print("Boundary value problem failed to converge.")
        self.q = self.solution.y[:len(initial_q_array[:,0])]
        self.qdot = self.solution.y[len(initial_q_array[:,0]):]
        self.t = self.solution.x

        return self.solution