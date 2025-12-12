import numpy as np
import cvxpy as cp
from control import dlqr
from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron 
class MPCControl_yvel(MPCControl_base):
    """MPC for y-velocity control (y-subsystem without position)."""
    # Reduced y-subsystem: [ω_x, α, v_y] (no position y)
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])  # δ₁
    
    def _setup_controller(self) -> None:
        """
        MPC for y-velocity subsystem.
        States: [ω_x, α, v_y]
        Input: δ₁
        Constraints: |α| ≤ 10°, |δ₁| ≤ 15°
        """
        N = self.N
        nx = self.nx
        nu = self.nu
        
        self.x_var = cp.Variable((nx, N + 1))   # Δx_k = [Δω_x, Δα, Δv_y]
        self.u_var = cp.Variable((nu, N))       # Δu_k = [Δδ₁]
        
        self.x0_par = cp.Parameter(nx)
        self.x_ref_par = cp.Parameter(nx)
        self.u_ref_par = cp.Parameter(nu)
        
        # Cost matrices - tune these for performance
        Q = np.diag([1.0, 50.0, 10.0])   # penalize α and v_y strongly
        R = np.diag([0.1])               # small penalty on δ₁
        K, P, _ = dlqr(self.A, self.B, Q, R)
        self.Q, self.R, self.P = Q, R, P
        
        
       
        # Terminal set 
    

        alpha_max = np.deg2rad(10.0)
        delta_max = np.deg2rad(15.0)

        alpha_eq = float(self.xs[1])   # equilibrium α (usually 0)
        delta_eq = float(self.us[0])   # equilibrium δ₁ (usually 0)

        # Terminal constraints are in Δx-space: H * Δx <= h
        # Δx = [Δω_x, Δα, Δv_y]
        #
        # State constraint: |α_eq + Δα| <= alpha_max
        Hx = np.array([
            [0.0,  1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        hx = np.array([
            alpha_max - alpha_eq,
            alpha_max + alpha_eq,
        ])

        # Input constraint under LQR: Δu = K Δx, true u = delta_eq + Δu
        # |delta_eq + KΔx| <= delta_max
        Hu = np.vstack([K, -K])
        hu = np.array([
            delta_max - delta_eq,
            delta_max + delta_eq,
        ]).flatten()

        H = np.vstack([Hx, Hu])
        h = np.hstack([hx, hu])

        h = np.asarray(h, dtype=float).flatten()
        H = np.asarray(H, dtype=float)
        self.terminal_set = Polyhedron.from_Hrep(H, h)
        
        
        
        constraints = []
        
        # Initial condition
        constraints += [self.x_var[:, 0] == self.x0_par]
        
        # State constraint on α (index 1 in reduced model): |α| ≤ 10°
        alpha_max = np.deg2rad(10.0)
        for k in range(N + 1):
            alpha_k = self.xs[1] + self.x_var[1, k]  # true α = α_s + Δα
            constraints += [
                alpha_k <= alpha_max,
                alpha_k >= -alpha_max,
            ]
        
        # Input constraint on δ₁: |δ₁| ≤ 15°
        delta_max = np.deg2rad(15.0)
        for k in range(N):
            u_true = self.us[0] + self.u_var[0, k]  # scalar + scalar
            constraints += [
                u_true <= delta_max,
                u_true >= -delta_max,
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