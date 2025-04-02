import os
import pickle
import numpy as np
from quadrotor_dynamics import QuadrotorDynamics
from quattro_ilqr_tf.quattro_ilqr_tf import iLQR_TF  # Provided iLQR implementation

class QuadrotorMPC:
    """
    Model Predictive Controller (MPC) using iLQR for a quadrotor system.
    This controller updates at each time step, solving iLQR with a receding horizon.
    """

    def __init__(self, horizon=30, dt=0.01, integration_method="rk4", transformer_model=None, log_filename="quad_ilqr_log.pkl"):
        """
        Initializes the MPC controller.

        Parameters:
        - horizon: int, Number of time steps in the MPC window.
        - dt: float, Discretization time step.
        - integration_method: str, Either "rk4" or "euler" for state propagation.
        - transformer_model: model or None, Optionally supply a transformer-based model.
        - log_filename: str, Path to the pickle file where iLQR logs will eventually be saved.
        """
        self.horizon = horizon
        self.dt = dt
        self.integration_method = integration_method
        self.log_filename = log_filename

        # Create quadrotor dynamics instance.
        self.dynamics = QuadrotorDynamics()

        # Set desired (reference) state.
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
        # For hover, we desire the quadrotor to be at x=0, y=0, z=0.5 with zero velocities and zero angles.
        self.x_ref = np.zeros(12)
        self.x_ref[2] = 0.5

        # Define cost matrices (tune these as needed)
        # For example, penalize position error, velocity, orientation, and angular rates.
        self.Q = np.diag([10.0, 10.0, 50.0,   1.0, 1.0, 1.0,   10.0, 10.0, 50.0,   1.0, 1.0, 1.0])
        self.R = np.diag([0.01, 0.01, 0.01, 0.01])
        self.Qf = np.diag([100.0, 100.0, 500.0,   10.0, 10.0, 10.0,   100.0, 100.0, 500.0,   10.0, 10.0, 10.0])
        
        # Parameters for the soft barrier penalty (to enforce u >= 0)
        self.alpha = 1000.0  # penalty weight
        self.beta = 10.0     # smoothness factor for softplus

        # Initial control sequence: for quadrotor, control is 4D (motor thrusts)
        self.u_init = [np.zeros(4) for _ in range(horizon)]

        # iLQR transformer model (if any)
        self.transformer_model = transformer_model
        self.en_tf = False
        if self.transformer_model is not None:
            self.en_tf = True

        # Create iLQR solver instance
        self.ilqr = iLQR_TF(
            dynamics=self.discrete_dynamics,
            cost=self.running_cost,
            cost_final=self.final_cost,
            x0=self.x_ref,  # initial state for iLQR
            u_init=self.u_init,
            horizon=horizon,
            tf=self.transformer_model,
            en_tf=self.en_tf
        )

    def discrete_dynamics(self, x, u):
        """
        Wraps the quadrotor dynamics to use the chosen integration method.
        """
        return self.dynamics.discrete_dynamics(x, u, self.dt, method=self.integration_method)

    def softplus(self, x, beta):
        """
        Smooth approximation of the positive part of -x:
          softplus(x) = (1/beta)*log(1+exp(beta*x))
        so that softplus(-u) approximates max(0, -u) in a smooth way.
        """
        return np.log1p(np.exp(beta * x)) / beta

    def running_cost(self, x, u):
        """
        Quadratic running cost for state and control, with a soft barrier penalty on negative controls.
        """
        dx = x - self.x_ref
        cost = dx.T @ self.Q @ dx + u.T @ self.R @ u
        
        # Compute a smooth penalty for any u_i < 0
        # Here we use softplus(-u) as a smooth approximation of max(0, -u)
        penalty = np.sum(self.softplus(-u, self.beta)**2)
        cost += self.alpha * penalty
        return cost

    def final_cost(self, x):
        """
        Terminal cost that heavily penalizes deviation from x_ref.
        """
        dx = x - self.x_ref
        return dx.T @ self.Qf @ dx

    def control_step(self, x_current):
        """
        Computes the optimal control for the given current state using iLQR.
        Accumulates the solver logs in memory.

        Parameters:
          - x_current: np.ndarray (12,), the current quadrotor state
                     [x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r].

        Returns:
          - (optimal_x_seq, optimal_u_seq): the iLQR-optimized state and control sequences.
        """
        # Update the initial state for iLQR
        self.ilqr.x0 = x_current

        # Solve iLQR for the current state
        optimal_u_seq, optimal_x_seq = self.ilqr.optimize(x_ref=self.x_ref, verbose=False)

        # Shift control sequence for warm start in the next time step.
        self.ilqr.u = optimal_u_seq[1:].copy()
        self.ilqr.u.append(optimal_u_seq[-1])  # hold last control input

        return optimal_x_seq, optimal_u_seq
