import numpy as np
from cartpole_dynamics import CartPoleDynamics
from quattro_ilqr_tf.quattro_ilqr_tf import iLQR_TF
from scipy.linalg import solve_discrete_are, inv
from typing import List, Tuple, Optional, Dict

######################################
# Controller Switcher for Blending Control
######################################
class ControllerSwitcher:
    """
    Computes a blending weight for switching between a primary nonlinear controller 
    (e.g., iLQR with/without transformer) and a linear LQR controller. The blending 
    factor is based on the norm of the error and, optionally, on the acceleration 
    (second derivative) of the error.
    """

    def __init__(
        self,
        epsilon_low: float = 0.05,
        epsilon_high: float = 0.2,
        epsilon_dd: float = 0.1,
        gamma: float = 10.0,
        use_sigmoid: bool = False,
    ):
        """
        Initialize the switcher with threshold and damping parameters.
        
        Parameters:
            epsilon_low (float): Lower bound for the error norm below which full LQR is used.
            epsilon_high (float): Upper bound for the error norm above which full primary control is used.
            epsilon_dd (float): Threshold for the error acceleration norm.
            gamma (float): Steepness parameter for the sigmoid damping (if used).
            use_sigmoid (bool): If True, use a sigmoid function for acceleration damping;
                                otherwise use linear clipping.
        """
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.epsilon_dd = epsilon_dd
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid

        # History of error vectors for approximating acceleration (finite differences)
        self.error_history: List[np.ndarray] = []

    def update_error(self, error: np.ndarray) -> None:
        """
        Update the error history with the current error vector.
        
        Parameters:
            error (np.ndarray): The current error (x - x_ref).
        """
        self.error_history.append(error)
        if len(self.error_history) > 3:
            self.error_history.pop(0)

    def compute_current_error_norm(self) -> float:
        """
        Compute the norm of the most recent error.
        
        Returns:
            float: The L2 norm of the current error.
        """
        if not self.error_history:
            return 0.0
        return np.linalg.norm(self.error_history[-1])

    def compute_acceleration_norm(self, dt: float) -> float:
        """
        Compute an approximate acceleration norm (second derivative) of the error using finite differences.
        
        Parameters:
            dt (float): Time step used for finite difference approximation.
        
        Returns:
            float: The L2 norm of the approximated acceleration. If insufficient history is available, returns 0.
        """
        if len(self.error_history) < 3:
            return 0.0
        e_prev, e_curr, e_next = self.error_history[0], self.error_history[1], self.error_history[2]
        # Approximate second derivative: e_ddot ~ (e_next - 2*e_curr + e_prev) / (dt^2)
        e_ddot = (e_next - 2 * e_curr + e_prev) / (dt ** 2)
        return np.linalg.norm(e_ddot)

    def get_blending_weight(self, dt: float) -> float:
        """
        Compute the final blending weight based solely on the error norm.
        (Acceleration-based weighting is currently commented out but can be re-enabled.)
        
        Returns:
            float: Blending weight in [0, 1] where 0 means full LQR control and 1 means full primary controller.
        """
        # Compute weight based on current error norm
        e_norm = self.compute_current_error_norm()
        if e_norm <= self.epsilon_low:
            w_e = 0.0
        elif e_norm >= self.epsilon_high:
            w_e = 1.0
        else:
            w_e = (e_norm - self.epsilon_low) / (self.epsilon_high - self.epsilon_low)

        # --- Uncomment below to include acceleration damping ---
        # acc_norm = self.compute_acceleration_norm(dt)
        # w_acc_linear = max(0.0, 1 - acc_norm / self.epsilon_dd)
        # if self.use_sigmoid:
        #     # Sigmoid damping factor: near 1 when acc_norm is below threshold, approaching 0 when above.
        #     w_acc = 1.0 / (1.0 + np.exp(self.gamma * (acc_norm - self.epsilon_dd)))
        # else:
        #     w_acc = w_acc_linear
        #
        # Final blending weight: product of error weight and acceleration weight.
        # w_final = w_e * w_acc

        # Currently, only error norm is used.
        w_final = w_e
        return w_final


######################################
# Cart-Pole Model Predictive Controller (MPC)
######################################
class CartPoleMPC:
    """
    Controller for the cart-pole system supporting multiple modes:
    
      1. LQR only,
      2. iLQR only (without transformer),
      3. iLQR + Transformer (blended with LQR via switching logic).
      
    Mode selection is done via the following flags (only one should be True; 
    if more than one is set, 'lqr_only' takes priority, then 'ilqr_only'):
    
      - lqr_only
      - ilqr_only
      - ilqr_tf_only
      - ilqr_tf_blend
      - use_transformer
      
    The default mode is iLQR-only if no flag is provided.
    """

    def __init__(
        self,
        horizon: int = 30,
        dt: float = 0.01,
        integration_method: str = "rk4",
        transformer_model: Optional[object] = None,
        log_filename: str = "ilqr_log.pkl",
        switcher_params: Optional[Dict] = None,
        lqr_only: bool = False,
        ilqr_only: bool = False,
        ilqr_tf_only: bool = False,
        ilqr_tf_blend: bool = False,
        use_transformer: bool = False,
    ):
        """
        Initialize the CartPoleMPC controller with cost parameters, dynamics, and mode selection.
        
        Parameters:
            horizon (int): Number of time steps in the receding horizon.
            dt (float): Discretization time step.
            integration_method (str): "rk4" or "euler" for state propagation.
            transformer_model (object, optional): Transformer model instance; can be None.
            log_filename (str): File path for saving iLQR logs.
            switcher_params (dict, optional): Dictionary of parameters for ControllerSwitcher.
            lqr_only (bool): If True, use only LQR control.
            ilqr_only (bool): If True, use only iLQR control (transformer disabled).
            ilqr_tf_only (bool): If True, use iLQR with transformer (no blending).
            ilqr_tf_blend (bool): If True, blend iLQR+Transformer with LQR.
            use_transformer (bool): Flag for transformer usage (provided for backward compatibility).
        """
        self.horizon = horizon
        self.dt = dt
        self.integration_method = integration_method
        self.log_filename = log_filename
        self.lqr_only = lqr_only
        self.ilqr_only = ilqr_only
        self.ilqr_tf_only = ilqr_tf_only
        self.ilqr_tf_blend = ilqr_tf_blend
        self.use_transformer = use_transformer

        # Create cart-pole dynamics and reference state.
        self.dynamics = CartPoleDynamics()
        self.x_ref = np.array([0.0, 0.0, 0.0, 0.0])

        # Define cost matrices for iLQR and LQR.
        self.Q = np.diag([5.0, 0.1, 10.0, 0.1])
        self.R = np.diag([0.001])
        self.Qf = np.diag([50.0, 6.0, 100.0, 0.1])
        self.Q_lqr = np.diag([1.0, 0.1, 10.0, 0.1])
        self.R_lqr = np.diag([0.001])

        # Initialize control input sequence.
        self.u_init: List[np.ndarray] = [np.array([0.0]) for _ in range(horizon)]

        # Select mode and set transformer model if applicable.
        if not self.lqr_only:
            if self.ilqr_only:
                tf_model = None  # Disable transformer for iLQR-only mode.
            elif self.ilqr_tf_only or self.ilqr_tf_blend:
                tf_model = transformer_model  # Enable transformer.
            else:
                # Default to iLQR-only if no mode flag is explicitly set.
                tf_model = None

            # Initialize the iLQR solver.
            self.ilqr = iLQR_TF(
                dynamics=self.discrete_dynamics,
                cost=self.running_cost,
                cost_final=self.final_cost,
                x0=self.x_ref,
                u_init=self.u_init,
                horizon=self.horizon,
                tf=tf_model,
                tol=1e-1,
            )
        else:
            self.ilqr = None  # Not used in LQR-only mode.

        # Initialize the controller switcher for blending mode.
        if switcher_params is None:
            switcher_params = {}
        self.switcher = ControllerSwitcher(
            epsilon_low=switcher_params.get("epsilon_low", 0.5),
            epsilon_high=switcher_params.get("epsilon_high", 1.5),
            epsilon_dd=switcher_params.get("epsilon_dd", 0.01),
            gamma=switcher_params.get("gamma", 10.0),
            use_sigmoid=switcher_params.get("use_sigmoid", False),
        )

    def discrete_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Propagate the state using the selected integration method.
        
        Parameters:
            x (np.ndarray): Current state.
            u (np.ndarray): Control input.
            
        Returns:
            np.ndarray: Next state after propagation.
        """
        return self.dynamics.discrete_dynamics(x, u, self.dt, method=self.integration_method)

    def running_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """
        Compute the running cost for state x and control u.
        
        Parameters:
            x (np.ndarray): Current state.
            u (np.ndarray): Control input.
            
        Returns:
            float: The cost computed as quadratic functions of state error and control effort.
        """
        dx = x - self.x_ref
        return float(dx.T @ self.Q @ dx + u.T @ self.R @ u)

    def final_cost(self, x: np.ndarray) -> float:
        """
        Compute the final cost for the terminal state.
        
        Parameters:
            x (np.ndarray): Terminal state.
            
        Returns:
            float: The terminal cost based on the state error.
        """
        dx = x - self.x_ref
        return float(dx.T @ self.Qf @ dx)

    def linearized_dynamics(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the discrete-time linearized dynamics (A_d, B_d) of the system.
        
        Parameters:
            dt (float): Time step for discretization.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Discrete-time state and input matrices.
        """
        A, B = self.dynamics.linearized_dynamics()
        n = A.shape[0]
        A_d = np.eye(n) + dt * A
        B_d = dt * B
        return A_d, B_d

    def compute_linear_lqr_control(self, x_current: np.ndarray) -> np.ndarray:
        """
        Compute the LQR control input using an infinite-horizon discrete-time formulation.
        
        Parameters:
            x_current (np.ndarray): The current state of the system.
            
        Returns:
            np.ndarray: The optimal control input computed from the LQR.
        """
        A_d, B_d = self.linearized_dynamics(self.dt)
        P = solve_discrete_are(A_d, B_d, self.Q_lqr, self.R_lqr)
        K = inv(self.R_lqr + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
        u_opt = -K @ (x_current - self.x_ref)
        return u_opt.flatten()

    def control_step(self, x_current: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute the control input for the current state based on the selected mode.
        
        Modes:
          - LQR-only: Directly compute the LQR control.
          - iLQR-only or iLQR + Transformer-only: Compute control from iLQR.
          - Blending: Use switching logic to blend between LQR and iLQR-based control.
          
        Parameters:
            x_current (np.ndarray): The current state of the system.
            
        Returns:
            Tuple[List[np.ndarray], np.ndarray]:
                - optimal_x_seq: Optimized state trajectory (empty list if not computed).
                - u_final: Final computed control input.
        """
        # --- LQR-only mode ---
        if self.lqr_only:
            u_final = -self.compute_linear_lqr_control(x_current)
            return [], u_final

        # --- iLQR-only or iLQR + Transformer-only mode ---
        if self.ilqr_only or self.ilqr_tf_only:
            self.ilqr.x0 = x_current
            optimal_u_seq, optimal_x_seq = self.ilqr.optimize(x_ref=self.x_ref)
            u_final = optimal_u_seq[0]
            # Update the control sequence for the next iteration.
            self.ilqr.u = optimal_u_seq[1:].copy() + [optimal_u_seq[-1]]
            return optimal_x_seq, u_final

        # --- Blending mode (iLQR + Transformer with LQR blending) ---
        # Update switching logic with current error.
        error = x_current - self.x_ref
        self.switcher.update_error(error)
        w = self.switcher.get_blending_weight(self.dt)

        if w <= 0.05:
            # Almost full LQR control.
            u_final = -self.compute_linear_lqr_control(x_current)
            optimal_x_seq = []
        elif w >= 0.95:
            # Almost full iLQR control.
            self.ilqr.x0 = x_current
            optimal_u_seq, optimal_x_seq = self.ilqr.optimize(x_ref=self.x_ref)
            u_final = optimal_u_seq[0]
            self.ilqr.u = optimal_u_seq[1:].copy() + [optimal_u_seq[-1]]
        else:
            # Blend between iLQR and LQR based on weight.
            self.ilqr.x0 = x_current
            optimal_u_seq, optimal_x_seq = self.ilqr.optimize(x_ref=self.x_ref)
            u_primary = optimal_u_seq[0]
            u_lqr = -self.compute_linear_lqr_control(x_current)
            u_final = w * u_primary + (1 - w) * u_lqr
            self.ilqr.u = optimal_u_seq[1:].copy() + [optimal_u_seq[-1]]

        return optimal_x_seq, u_final
