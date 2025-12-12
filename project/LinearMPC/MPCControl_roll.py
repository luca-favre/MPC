import numpy as np
import cvxpy as cp
from control import dlqr
from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron
class MPCControl_roll(MPCControl_base):
    """MPC for roll control."""
    # Roll subsystem: [ω_z, γ]
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])  # P_diff
    
    def _setup_controller(self) -> None:
        """
        MPC for roll dynamics.
        States: [ω_z, γ]
        Input: P_diff
        Constraints:  |P_diff| ≤ 20
        """
        N = self.N
        nx = self.nx
        nu = self.nu
        
        self.x_var = cp.Variable((nx, N + 1))   # Δx_k = [Δω_z, Δγ]
        self.u_var = cp.Variable((nu, N))       # Δu_k = [ΔP_diff]
        
        self.x0_par = cp.Parameter(nx)
        self.x_ref_par = cp.Parameter(nx)
        self.u_ref_par = cp.Parameter(nu)
        
        # Cost matrices - tune these for performance
        Q = np.diag([1.0, 50.0])   # stronger weight on γ
        R = np.diag([0.1])         # penalty on P_diff
        # IMPORTANT: dlqr returns u = -K_lqr x
        K_lqr, P, _ = dlqr(self.A, self.B, Q, R)
        K = -K_lqr  # so we can write Δu = K Δx
        self.K = K
        self.Q, self.R, self.P = Q, R, P

        # -------------------------------
        # Terminal invariant set Xf (LQR-based)
        # -------------------------------
        Pdiff_max = 20.0
        Pdiff_eq = float(self.us[0])

        # No explicit state constraints on (ωz, γ) in the project.
        # We still need a bounded X to build a Polyhedron for the invariant-set iteration.
        wz_bound = 10.0
        gamma_bound = np.pi
        X = Polyhedron.from_Hrep(
            np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            np.array([wz_bound, wz_bound, gamma_bound, gamma_bound])
        )

        # U: bounds on ΔPdiff since u_true = Pdiff_eq + Δu
        U = Polyhedron.from_Hrep(
            np.array([[1.0], [-1.0]]),
            np.array([Pdiff_max - Pdiff_eq, Pdiff_max + Pdiff_eq])
        )

        KU = Polyhedron.from_Hrep(U.A @ K, U.b)
        X_and_KU = X.intersect(KU)

        Acl = self.A + self.B @ K
        self.Xf = self.max_invariant_set(Acl, X_and_KU)
        constraints = []
        
        # Initial condition
        constraints += [self.x_var[:, 0] == self.x0_par]
        
        Pdiff_max = 20.0
        for k in range(N):
            u_true = self.us[0] + self.u_var[0, k]  # scalar + scalar
            constraints += [
                u_true <= Pdiff_max,
                u_true >= -Pdiff_max,
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