import numpy as np
import cvxpy as cp
from control import dlqr
from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron 
class MPCControl_zvel(MPCControl_base):
    """MPC for vertical velocity control (z-subsystem without position)."""
    x_ids: np.ndarray = np.array([8])  # v_z only (no position z)
    u_ids: np.ndarray = np.array([2])  # P_avg
    
    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float
    
    def _setup_controller(self) -> None:
        """
        MPC for vertical velocity v_z, input P_avg.
        Constraints: 40 ≤ P_avg ≤ 80
        """
        N = self.N
        nx = self.nx
        nu = self.nu
        
        self.x_var = cp.Variable((nx, N + 1))   # Δv_z
        self.u_var = cp.Variable((nu, N))       # ΔP_avg
        
        self.x0_par = cp.Parameter(nx)
        self.x_ref_par = cp.Parameter(nx)
        self.u_ref_par = cp.Parameter(nu)
        
        # Cost matrices - tune these for performance
        Q = np.array([[10.0]])   # penalize v_z deviation
        R = np.array([[0.1]])    # penalize throttle deviation
        # IMPORTANT: dlqr returns u = -K_lqr x
        K_lqr, P, _ = dlqr(self.A, self.B, Q, R)
        K = -K_lqr  # so we can write Δu = K Δx
        self.K = K
        self.Q, self.R, self.P = Q, R, P

        # -------------------------------
        # Terminal invariant set Xf (LQR-based)
        # -------------------------------
        P_min = 40.0
        P_max = 80.0
        P_eq = float(self.us[0])

        # For 1D we still need some (non-active) state bounds to build a Polyhedron.
        # Choose a wide bound so it doesn't bind; the invariant set will be shaped by U via K.
        vz_bound = 30.0
        X = Polyhedron.from_Hrep(np.array([[1.0], [-1.0]]), np.array([vz_bound, vz_bound]))

        # U: bounds on ΔPavg since u_true = P_eq + Δu
        U = Polyhedron.from_Hrep(
            np.array([[1.0], [-1.0]]),
            np.array([P_max - P_eq, P_eq - P_min])
        )

        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        X_and_KU = X.intersect(KU)

        Acl = self.A + self.B @ K
        self.Xf = self.max_invariant_set(Acl, X_and_KU)
        constraints = []
        
        # Initial condition
        constraints += [self.x_var[:, 0] == self.x0_par]
        
        # Input constraint: 40 ≤ P_avg ≤ 80 (true input = us + Δu)
        P_min = 40.0
        P_max = 80.0
        for k in range(N):
            u_true = self.us[0] + self.u_var[0, k]  # scalar + scalar
            constraints += [
                u_true >= P_min,
                u_true <= P_max,
            ]
        
        # Dynamics: Δx_{k+1} = A*Δx_k + B*Δu_k
        for k in range(N):
            constraints += [
                self.x_var[:, k + 1] ==
                self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
            ]
        
        # Objective: stage cost + terminal cost
        cost = 0
        for k in range(N):
            dx_k = self.x_var[:, k] - self.x_ref_par
            du_k = self.u_var[:, k] - self.u_ref_par
            cost += cp.quad_form(dx_k, Q) + cp.quad_form(du_k, R)
        
        dx_N = self.x_var[:, N] - self.x_ref_par
        cost += cp.quad_form(dx_N, P)

        # Terminal constraint x_N ∈ Xf (recursive feasibility)
        constraints += [self.Xf.A @ self.x_var[:, N] <= self.Xf.b]
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        
        # Part 5 placeholders
        self.d_estimate = np.zeros((1,))
        self.d_gain = 0.0
    
    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        """FOR PART 5 OF THE PROJECT – dummy implementation"""
        pass
    
    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve MPC and return control input.
        
        Returns:
            u0: Control input for this subsystem (shape: nu)
            x_traj: Predicted state trajectory in true coordinates (shape: nx x N+1)
            u_traj: Predicted input trajectory in true coordinates (shape: nu x N)
        """
        if x_target is None:
            x_target = self.xs.copy()
        else:
            x_target = x_target.copy()
        
        if u_target is None:
            u_target = self.us.copy()
        else:
            u_target = u_target.copy()
        
        # Convert to deviation coordinates
        dx0 = x0 - self.xs
        dx_ref = x_target - self.xs
        du_ref = u_target - self.us
        
        # Set parameters
        self.x0_par.value = dx0
        self.x_ref_par.value = dx_ref
        self.u_ref_par.value = du_ref
        
        # Solve MPC
        self.ocp.solve(solver=cp.OSQP, warm_start=True)
        
        # Fallback if solver fails
        if self.ocp.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            u0 = self.us.copy()
            x_traj = np.tile(self.xs.reshape(-1, 1), (1, self.N + 1))
            u_traj = np.tile(self.us.reshape(-1, 1), (1, self.N))
            return u0, x_traj, u_traj
        
        # Extract optimal trajectories in deviation coordinates
        dx_traj = self.x_var.value
        du_traj = self.u_var.value
        
        # Convert back to true coordinates
        x_traj = dx_traj + self.xs.reshape(-1, 1)
        u_traj = du_traj + self.us.reshape(-1, 1)
        
        # First control input
        u0 = u_traj[:, 0]
        
        return u0, x_traj, u_traj