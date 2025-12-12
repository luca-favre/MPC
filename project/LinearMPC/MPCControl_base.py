import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    """Base MPC class on a reduced subsystem."""

    # To be set in subclasses
    x_ids: np.ndarray
    u_ids: np.ndarray

    # Optimization system
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    # Optimization problem
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition: reduced subsystem (continuous → discrete)
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        """
        Generic, unconstrained MPC setup on the reduced subsystem.

        Subclasses can override this method to add constraints or change Q/R.
        """
        N = self.N
        nx = self.nx
        nu = self.nu

        # Decision variables (deviation coordinates Δx, Δu)
        self.x_var = cp.Variable((nx, N + 1))  # Δx_k
        self.u_var = cp.Variable((nu, N))      # Δu_k

        # Parameters: initial state and references in deviation coordinates
        self.x0_par = cp.Parameter(nx)         # Δx_0
        self.x_ref_par = cp.Parameter(nx)      # Δx_ref
        self.u_ref_par = cp.Parameter(nu)      # Δu_ref

        # Default quadratic costs (can be overridden by subclasses)
        Q = np.eye(nx)
        R = 0.1 * np.eye(nu)
        _, P, _ = dlqr(self.A, self.B, Q, R)

        self.Q = Q
        self.R = R
        self.P = P

       
        constraints = []

        # Initial condition
        constraints += [self.x_var[:, 0] == self.x0_par]

        # Dynamics
        for k in range(N):
            constraints += [
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
            ]

        # Objective: stage + terminal cost in deviation coordinates
        cost = 0
        for k in range(N):
            dx_k = self.x_var[:, k] - self.x_ref_par
            du_k = self.u_var[:, k] - self.u_ref_par
            cost += cp.quad_form(dx_k, Q) + cp.quad_form(du_k, R)

        dx_N = self.x_var[:, N] - self.x_ref_par
        cost += cp.quad_form(dx_N, P)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generic MPC call:

        - Works in deviation coordinates around (xs, us)
        - Solves the CVXPY problem self.ocp
        - Returns first control input and predicted trajectories in TRUE coordinates.
        """
        # Default regulation to trim (xs, us) if no target provided
        if x_target is None:
            x_target = self.xs.copy()
        else:
            x_target = x_target.copy()

        if u_target is None:
            u_target = self.us.copy()
        else:
            u_target = u_target.copy()

        # Deviation coordinates
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

        # Optimal deviation trajectories
        dx_traj = self.x_var.value  # (nx, N+1)
        du_traj = self.u_var.value  # (nu, N)

        # Convert back to TRUE states and inputs
        x_traj = dx_traj + self.xs.reshape(-1, 1)
        u_traj = du_traj + self.us.reshape(-1, 1)

        # First control input for this subsystem
        u0 = u_traj[:, 0]

        return u0, x_traj, u_traj


    # ------------------------------------------------------------------
    # Terminal set utilities (Deliverable 3.1 / 3.2)
    # ------------------------------------------------------------------
    @staticmethod
    def max_invariant_set(A_cl: np.ndarray, X: Polyhedron, max_iter: int = 50) -> Polyhedron:
        """Compute the maximal positively invariant set for x^+ = A_cl x, inside X.

        Implements the standard fixed-point iteration:
            O_{k+1} = O_k ∩ Pre(O_k),  Pre(O) = {x | A_cl x ∈ O}

        Notes:
        - X and the returned set are in the same coordinates (here: deviation coordinates).
        - Equality check relies on mpt4py Polyhedron __eq__.
        """
        O = Polyhedron.from_Hrep(X.A, X.b)
        itr = 0
        while itr < max_iter:
            O_prev = O
            F, f = O.A, O.b
            O_pre = Polyhedron.from_Hrep(F @ A_cl, f)
            O = O.intersect(O_pre)
            if O == O_prev:
                break
            itr += 1
        return O




