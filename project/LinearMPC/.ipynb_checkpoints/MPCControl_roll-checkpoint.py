import numpy as np
import cvxpy as cp
from control import dlqr
from .MPCControl_base import MPCControl_base

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
        Constraints: |γ| ≤ 10° (optional, linearization valid for any γ), |P_diff| ≤ 40
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
        _, P, _ = dlqr(self.A, self.B, Q, R)
        self.Q, self.R, self.P = Q, R, P
        
        constraints = []
        
        # Initial condition
        constraints += [self.x_var[:, 0] == self.x0_par]
        
        # # Roll angle constraint |γ| ≤ 10° 
        # gamma_max = np.deg2rad(10.0)
        # for k in range(N + 1):
        #     gamma_k = self.xs[1] + self.x_var[1, k]  # true γ = γ_s + Δγ
        #     constraints += [
        #         gamma_k <= gamma_max,
        #         gamma_k >= -gamma_max,
        #     ]
        
        # Input constraint on P_diff: |P_diff| ≤ 40
        Pdiff_max = 40.0
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