#!/usr/bin/env python3
"""
quattro_ilqr_tf: A Hybrid iLQR and Transformer-based Optimization Framework

Author: Your Name
License: MIT
"""

import time
import functools
import numpy as np


def measure_time(record_attr_name):
    """
    Decorator factory that records the runtime of a method in a named list attribute on the instance.

    Args:
        record_attr_name (str): Name of the list attribute on the instance to store timings.

    Returns:
        decorator (function): Decorator that appends execution time to the specified instance list.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            elapsed_time = time.time() - start_time
            getattr(self, record_attr_name).append(elapsed_time)
            return result
        return wrapper
    return decorator


def measure_time_with_list(record_list):
    """
    Decorator factory that records the runtime of a function into a provided mutable list.

    Args:
        record_list (list): A list to which execution times are appended.

    Returns:
        decorator (function): Decorator that appends execution time to record_list.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            record_list.append(elapsed_time)
            return result
        return wrapper
    return decorator


class iLQR_TF:
    """
    A hybrid iLQR and Transformer-based optimization class.

    This class implements iterative LQR for trajectory optimization. Optionally,
    a transformer model can be used to predict controls within a specified window.
    """

    def __init__(
        self,
        dynamics,
        cost,
        cost_final,
        x0,
        u_init,
        horizon,
        dt=0.01,
        max_iter=100,
        tol=1e-3,
        tf=None,
        tf_window=10,
        blend_mode=False,
        enable_log=True
    ):
        """
        Initialize iLQR_TF.

        Args:
            dynamics (callable): System dynamics function f(x, u) -> x_next.
            cost (callable): Running cost function L(x, u) -> float.
            cost_final (callable): Final cost function Lf(x) -> float.
            x0 (np.ndarray): Initial state vector.
            u_init (list[np.ndarray]): Initial control sequence (length = horizon).
            horizon (int): Number of time steps in the optimization.
            dt (float): Time step size.
            max_iter (int): Maximum number of iLQR iterations.
            tol (float): Convergence tolerance on cost improvement.
            tf (object or None): Transformer model wrapper with a .predict(x_seq, prompt) method.
            tf_window (int): Window size (time steps) for which the Transformer predicts controls.
                             Must be less than the horizon.
            blend_mode (bool): If True, iLQR uses both LQR backward pass and Transformer in synergy.
            enable_log (bool): If False, iteration logs (self.logs) won't be stored to save memory.
        """
        self.f = dynamics
        self.L = cost
        self.Lf = cost_final
        self.x0 = x0
        self.u = u_init
        self.horizon = horizon
        self.dt = dt
        self.total_iter = 0
        self.max_iter = max_iter
        self.tol = tol

        self.state_offset = np.zeros_like(self.x0)

        # Validate tf_window
        if tf_window >= horizon:
            raise ValueError("tf_window must be less than the horizon.")
        self.tf_window = tf_window

        # Transformer usage
        self.tf = tf
        self.blend_mode = blend_mode

        # Logging
        self.enable_log = enable_log
        self.logs = []
        self.log_the_optimal_solution = False

        # Timing lists
        self.backward_pass_time = []
        self.forward_pass_time = []
        self.total_time = []
        self.inference_time = []

        # Decorate transformer's predict if provided
        if self.tf is not None:
            # Force the window to match the transformer's prompt_len if that is required
            if hasattr(self.tf, 'prompt_len'):
                self.tf_window = self.tf.prompt_len

            # Decorate the tf.predict method for timing
            self.tf.predict = measure_time_with_list(self.inference_time)(self.tf.predict)

    # -----------------------------------------------------------------------
    # Public Methods: Simulation, Cost Evaluation, Finite Difference Helpers
    # -----------------------------------------------------------------------
    def simulate(self, u_seq):
        """
        Simulate the system forward in time using the current control sequence.

        Args:
            u_seq (list[np.ndarray]): Control sequence, length == horizon.

        Returns:
            x_seq (np.ndarray): State trajectory of shape (horizon + 1, state_dim).
        """
        x_seq = [self.x0]
        for u in u_seq:
            x_next = self.f(x_seq[-1], u)
            x_seq.append(x_next)
        return np.array(x_seq)

    def compute_total_cost(self, x_seq, u_seq):
        """
        Compute the total cost (running + final) for a given (x_seq, u_seq).

        Args:
            x_seq (np.ndarray): State trajectory of shape (horizon + 1, state_dim).
            u_seq (list[np.ndarray]): Control sequence of shape (horizon, control_dim).

        Returns:
            float: Total cost over the trajectory.
        """
        cost_val = 0.0
        for t in range(self.horizon):
            cost_val += self.L(x_seq[t], u_seq[t])
        cost_val += self.Lf(x_seq[-1])
        return cost_val

    # ---------------------
    # Finite Difference Tools
    # ---------------------
    def _finite_diff_gradient_final(self, x, eps=1e-5):
        """
        Compute gradient of the final cost w.r.t. state x via finite differences.

        Args:
            x (np.ndarray): State vector.
            eps (float): Small step for finite differences.

        Returns:
            grad (np.ndarray): Gradient of Lf(x).
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = np.copy(x)
            x2 = np.copy(x)
            x1[i] += eps
            x2[i] -= eps
            grad[i] = (self.Lf(x1) - self.Lf(x2)) / (2 * eps)
        return grad

    def _finite_diff_hessian_final(self, x, eps=1e-5):
        """
        Compute Hessian of the final cost w.r.t. state x via finite differences.

        Args:
            x (np.ndarray): State vector.
            eps (float): Small step for finite differences.

        Returns:
            hess (np.ndarray): Hessian of Lf(x), shape (len(x), len(x)).
        """
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_pp = np.copy(x); x_pp[i] += eps; x_pp[j] += eps
                x_pm = np.copy(x); x_pm[i] += eps; x_pm[j] -= eps
                x_mp = np.copy(x); x_mp[i] -= eps; x_mp[j] += eps
                x_mm = np.copy(x); x_mm[i] -= eps; x_mm[j] -= eps
                hess[i, j] = (self.Lf(x_pp) - self.Lf(x_pm)
                              - self.Lf(x_mp) + self.Lf(x_mm)) / (4 * eps ** 2)
        return hess

    def _compute_dynamics_jacobians(self, x, u, eps=1e-5):
        """
        Compute Jacobians of the dynamics (A, B) at a given (x, u) via finite differences.

        A = d f(x,u)/dx,  B = d f(x,u)/du

        Args:
            x (np.ndarray): State vector.
            u (np.ndarray): Control vector.
            eps (float): Small step for finite differences.

        Returns:
            A (np.ndarray): Jacobian of shape (state_dim, state_dim).
            B (np.ndarray): Jacobian of shape (state_dim, control_dim).
        """
        n = x.shape[0]
        m = u.shape[0]
        A = np.zeros((n, n))
        B = np.zeros((n, m))

        # Compute A
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            f_plus = self.f(x + dx, u)
            f_minus = self.f(x - dx, u)
            A[:, i] = (f_plus - f_minus) / (2 * eps)

        # Compute B
        for i in range(m):
            du = np.zeros(m)
            du[i] = eps
            f_plus = self.f(x, u + du)
            f_minus = self.f(x, u - du)
            B[:, i] = (f_plus - f_minus) / (2 * eps)

        return A, B

    def _compute_cost_derivatives(self, x, u, eps=1e-5):
        """
        Compute cost and its derivatives up to second order via finite differences.

        Returns:
            L_val (float): The running cost at (x, u).
            L_x  (np.ndarray): Derivative of cost w.r.t. x.
            L_u  (np.ndarray): Derivative of cost w.r.t. u.
            L_xx (np.ndarray): Second derivative w.r.t. x.
            L_uu (np.ndarray): Second derivative w.r.t. u.
            L_xu (np.ndarray): Mixed second derivative w.r.t. x and u.
        """
        n = x.shape[0]
        m = u.shape[0]

        # Base cost
        L_val = self.L(x, u)

        # First derivatives
        L_x = np.zeros(n)
        L_u = np.zeros(m)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            L_x[i] = (self.L(x + dx, u) - self.L(x - dx, u)) / (2 * eps)
        for i in range(m):
            du = np.zeros(m)
            du[i] = eps
            L_u[i] = (self.L(x, u + du) - self.L(x, u - du)) / (2 * eps)

        # Second derivatives
        L_xx = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dx_i = np.zeros(n); dx_j = np.zeros(n)
                dx_i[i] = eps; dx_j[j] = eps
                L_xx[i, j] = (
                    self.L(x + dx_i + dx_j, u)
                    - self.L(x + dx_i - dx_j, u)
                    - self.L(x - dx_i + dx_j, u)
                    + self.L(x - dx_i - dx_j, u)
                ) / (4 * eps**2)

        L_uu = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                du_i = np.zeros(m); du_j = np.zeros(m)
                du_i[i] = eps; du_j[j] = eps
                L_uu[i, j] = (
                    self.L(x, u + du_i + du_j)
                    - self.L(x, u + du_i - du_j)
                    - self.L(x, u - du_i + du_j)
                    + self.L(x, u - du_i - du_j)
                ) / (4 * eps**2)

        L_xu = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dx = np.zeros(n)
                du = np.zeros(m)
                dx[i] = eps
                du[j] = eps
                L_xu[i, j] = (
                    self.L(x + dx, u + du)
                    - self.L(x + dx, u - du)
                    - self.L(x - dx, u + du)
                    + self.L(x - dx, u - du)
                ) / (4 * eps**2)

        return L_val, L_x, L_u, L_xx, L_uu, L_xu

    # ------------------------------
    # Backward / Forward Pass Methods
    # ------------------------------
    @measure_time("backward_pass_time")
    def backward_pass(self, x_seq, u_seq):
        """
        Standard iLQR backward pass over the entire horizon.

        Args:
            x_seq (np.ndarray): State trajectory of shape (horizon+1, state_dim).
            u_seq (list[np.ndarray]): Control sequence of shape (horizon, control_dim).

        Returns:
            k_seq (list[np.ndarray]): Feedforward terms, length = horizon.
            K_seq (list[np.ndarray]): Feedback gains, length = horizon.
        """
        x_final = x_seq[-1]
        V_x = self._finite_diff_gradient_final(x_final)
        V_xx = self._finite_diff_hessian_final(x_final)

        k_seq = [None] * self.horizon
        K_seq = [None] * self.horizon

        for t in reversed(range(self.horizon)):
            x = x_seq[t]
            u = u_seq[t]
            A, B = self._compute_dynamics_jacobians(x, u)
            _, L_x, L_u, L_xx, L_uu, L_xu = self._compute_cost_derivatives(x, u)
            L_ux = L_xu.T

            Q_x = L_x + A.T @ V_x
            Q_u = L_u + B.T @ V_x
            Q_xx = L_xx + A.T @ V_xx @ A
            Q_ux = L_ux + B.T @ V_xx @ A
            Q_uu = L_uu + B.T @ V_xx @ B

            # Regularize Q_uu
            reg = 1e-6 * np.eye(Q_uu.shape[0])
            Q_uu_reg = Q_uu + reg
            inv_Q_uu = np.linalg.inv(Q_uu_reg)

            k = -inv_Q_uu @ Q_u
            K = -inv_Q_uu @ Q_ux

            k_seq[t] = k
            K_seq[t] = K

            # Update cost-to-go
            V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            V_xx = 0.5 * (V_xx + V_xx.T)  # Ensure symmetry

        return k_seq, K_seq

    @measure_time("backward_pass_time")
    def backward_pass_segment(self, x_seq, u_seq, start_idx):
        """
        Compute a backward pass over a segment from 'start_idx' to horizon-1.

        Args:
            x_seq (np.ndarray): Full state trajectory of shape (horizon+1, state_dim).
            u_seq (list[np.ndarray]): Full control sequence of shape (horizon, control_dim).
            start_idx (int): Segment start index.

        Returns:
            k_seq_seg (list[np.ndarray]): Feedforward terms for the segment.
            K_seq_seg (list[np.ndarray]): Feedback gains for the segment.
        """
        x_final = x_seq[-1]
        V_x = self._finite_diff_gradient_final(x_final)
        V_xx = self._finite_diff_hessian_final(x_final)

        seg_length = self.horizon - start_idx
        k_seq_seg = [None] * seg_length
        K_seq_seg = [None] * seg_length

        for t in reversed(range(start_idx, self.horizon)):
            x = x_seq[t]
            u = u_seq[t]
            A, B = self._compute_dynamics_jacobians(x, u)
            _, L_x, L_u, L_xx, L_uu, L_xu = self._compute_cost_derivatives(x, u)
            L_ux = L_xu.T

            Q_x = L_x + A.T @ V_x
            Q_u = L_u + B.T @ V_x
            Q_xx = L_xx + A.T @ V_xx @ A
            Q_ux = L_ux + B.T @ V_xx @ A
            Q_uu = L_uu + B.T @ V_xx @ B

            # Regularize Q_uu
            reg = 1e-6 * np.eye(Q_uu.shape[0])
            Q_uu_reg = Q_uu + reg
            inv_Q_uu = np.linalg.inv(Q_uu_reg)

            k = -inv_Q_uu @ Q_u
            K = -inv_Q_uu @ Q_ux

            idx_seg = t - start_idx
            k_seq_seg[idx_seg] = k
            K_seq_seg[idx_seg] = K

            # Update cost-to-go
            V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            V_xx = 0.5 * (V_xx + V_xx.T)

        return k_seq_seg, K_seq_seg

    @measure_time("forward_pass_time")
    def forward_pass(self, x_seq, u_seq, k_seq, K_seq, alpha=1.0):
        """
        Standard iLQR forward pass over the entire horizon.

        Args:
            x_seq (np.ndarray): State trajectory from previous iteration (horizon+1, state_dim).
            u_seq (list[np.ndarray]): Control sequence from previous iteration (horizon, control_dim).
            k_seq (list[np.ndarray]): Feedforward terms from backward pass.
            K_seq (list[np.ndarray]): Feedback gains from backward pass.
            alpha (float): Step size (line search parameter).

        Returns:
            new_x_seq (np.ndarray): Updated state trajectory (horizon+1, state_dim).
            new_u_seq (list[np.ndarray]): Updated control sequence (horizon, control_dim).
            total_cost (float): Total cost of the updated trajectory.
        """
        new_u_seq = []
        new_x_seq = [self.x0]
        for t in range(self.horizon):
            dx = new_x_seq[t] - x_seq[t]
            du = k_seq[t] + K_seq[t] @ dx
            new_u = u_seq[t] + alpha * du
            new_u_seq.append(new_u)
            x_next = self.f(new_x_seq[t], new_u)
            new_x_seq.append(x_next)

        new_x_seq = np.array(new_x_seq)
        total_cost = self.compute_total_cost(new_x_seq, new_u_seq)
        return new_x_seq, new_u_seq, total_cost

    @measure_time("forward_pass_time")
    def forward_pass_segment(self, x_seq, u_seq, k_seq_seg, K_seq_seg, start_idx, alpha=1.0):
        """
        Forward pass over a segment from 'start_idx' to horizon-1.

        Args:
            x_seq (np.ndarray): Full state trajectory from previous iteration (horizon+1, state_dim).
            u_seq (list[np.ndarray]): Full control sequence from previous iteration (horizon, control_dim).
            k_seq_seg (list[np.ndarray]): Segment's feedforward terms.
            K_seq_seg (list[np.ndarray]): Segment's feedback gains.
            start_idx (int): Segment start index.
            alpha (float): Step size (line search parameter).

        Returns:
            new_x_seq_seg (np.ndarray): Updated sub-trajectory of shape (seg_length+1, state_dim).
            new_u_seq_seg (list[np.ndarray]): Updated sub-control sequence of length seg_length.
            seg_cost (float): Total cost of the updated segment.
        """
        new_u_seq_seg = []
        # Starting state is the state at 'start_idx' from x_seq
        new_x_seq_seg = [x_seq.tolist()[start_idx]]
        for t in range(start_idx, self.horizon):
            idx = t - start_idx
            dx = new_x_seq_seg[idx] - x_seq[t]
            du = k_seq_seg[idx] + K_seq_seg[idx] @ dx
            new_u = u_seq[t] + alpha * du
            new_u_seq_seg.append(new_u)
            x_next = self.f(new_x_seq_seg[idx], new_u)
            new_x_seq_seg.append(x_next)

        new_x_seq_seg = np.array(new_x_seq_seg)
        seg_cost = self.compute_total_cost(new_x_seq_seg, new_u_seq_seg)
        return new_x_seq_seg, new_u_seq_seg, seg_cost

    # ---------------------------------
    # Main Optimization Method
    # ---------------------------------
    @measure_time("total_time")
    def optimize(self, x_ref, verbose=False):
        """
        Perform the iLQR-based optimization, optionally using a Transformer model.

        If self.tf is None, standard iLQR runs over the entire horizon.
        Otherwise, the Transformer is used to predict a portion of the controls
        near the end of the horizon (or merges the Transformer + LQR results if blend_mode=True).

        Args:
            x_ref (np.ndarray): Reference state for error-based computations (user-defined usage).
            verbose (bool): If True, prints iteration information.

        Returns:
            (u_seq, final_x_seq): The optimized control sequence and corresponding state trajectory.
        """
        # 1) If no transformer provided, run pure iLQR
        if self.tf is None:
            u_seq = self.u
            for iteration in range(self.max_iter):
                x_seq = self.simulate(u_seq)
                current_cost = self.compute_total_cost(x_seq, u_seq)
                k_seq, K_seq = self.backward_pass(x_seq, u_seq)

                found_update = False
                chosen_alpha = None
                new_x_seq = None
                new_u_seq = None
                new_cost = None

                # Simple line search
                for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
                    cand_x_seq, cand_u_seq, cand_cost = self.forward_pass(
                        x_seq, u_seq, k_seq, K_seq, alpha
                    )
                    if cand_cost <= current_cost:
                        found_update = True
                        chosen_alpha = alpha
                        new_x_seq = cand_x_seq
                        new_u_seq = cand_u_seq
                        new_cost = cand_cost
                        u_seq = cand_u_seq
                        break

                if self.enable_log:
                    self.logs.append({
                        'iteration': iteration,
                        'x_seq': x_seq,
                        'u_seq': u_seq,
                        'current_cost': current_cost,
                        'k_seq': k_seq,
                        'K_seq': K_seq,
                        'alpha': chosen_alpha,
                        'new_x_seq': new_x_seq,
                        'new_u_seq': new_u_seq,
                        'new_cost': new_cost,
                        'found_update': found_update
                    })

                self.total_iter = iteration
                if verbose:
                    print(f"Iteration {iteration}, Cost: {new_cost:.4f}, Alpha: {chosen_alpha}")

                if (not found_update) or (abs(current_cost - new_cost) < self.tol):
                    if verbose:
                        print("Convergence achieved or no improvement (no TF).")
                    break

            self.u = u_seq
            final_x_seq = self.simulate(u_seq)
            return u_seq, final_x_seq

        # 2) Hybrid iLQR + Transformer
        else:
            u_seq = self.u
            for iteration in range(self.max_iter):
                x_seq = self.simulate(u_seq)
                current_cost = self.compute_total_cost(x_seq, u_seq)

                # Starting index for the Transformer window
                start_idx = self.horizon - self.tf_window

                if self.blend_mode:
                    # LQR tail segment
                    k_seq_seg, K_seq_seg = self.backward_pass_segment(x_seq, u_seq, start_idx)

                    # Flatten
                    dim_u = k_seq_seg[0].shape[0]
                    dim_x = K_seq_seg[0].shape[1]
                    k_seq_arr = np.array(k_seq_seg)
                    K_seq_arr = np.array(K_seq_seg)

                    K_seq_flat = K_seq_arr.reshape(k_seq_arr.shape[0], dim_u * dim_x)
                    kK_prompt = np.concatenate([k_seq_arr, K_seq_flat], axis=-1)

                    x_seq_err = x_seq.copy() - x_ref + self.state_offset

                    # Transformer inference
                    predicted_kK_flat = self.tf.predict(x_seq_err, kK_prompt)

                    # Reshape => (target_len, dim_u, 1 + dim_x)
                    target_len = predicted_kK_flat.shape[0]
                    predicted_kK = predicted_kK_flat.reshape(target_len, dim_u, 1 + dim_x)

                    predicted_k = predicted_kK[:, :, 0]
                    predicted_K = predicted_kK[:, :, 1:]

                    # Concatenate predicted front + LQR tail
                    full_k_seq = np.concatenate([predicted_k, k_seq_arr], axis=0)
                    full_K_seq = np.concatenate([predicted_K, K_seq_arr], axis=0)

                else:
                    # Single-branch approach
                    k_seq_seg, K_seq_seg = self.backward_pass_segment(x_seq, u_seq, start_idx)

                    dim_u = k_seq_seg[0].shape[0]
                    dim_x = K_seq_seg[0].shape[1]
                    k_seq_arr = np.array(k_seq_seg)
                    K_seq_arr = np.array(K_seq_seg)

                    K_seq_flat = K_seq_arr.reshape(k_seq_arr.shape[0], dim_u * dim_x)
                    kK_prompt = np.concatenate([k_seq_arr, K_seq_flat], axis=-1)

                    x_seq_err = x_seq - x_ref + self.state_offset

                    predicted_kK_flat = self.tf.predict(x_seq_err, kK_prompt)

                    target_len = predicted_kK_flat.shape[0]
                    predicted_kK = predicted_kK_flat.reshape(target_len, dim_u, 1 + dim_x)

                    predicted_k = predicted_kK[:, :, 0]
                    predicted_K = predicted_kK[:, :, 1:]

                    full_k_seq = np.concatenate([predicted_k, k_seq_arr], axis=0)
                    full_K_seq = np.concatenate([predicted_K, K_seq_arr], axis=0)

                # Line search
                found_update = False
                chosen_alpha = None
                new_x_seq = None
                new_u_seq = None
                new_cost = None

                for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
                    cand_x_seq, cand_u_seq, cand_cost = self.forward_pass(
                        x_seq, u_seq, full_k_seq, full_K_seq, alpha
                    )
                    if cand_cost <= current_cost:
                        found_update = True
                        chosen_alpha = alpha
                        new_x_seq = cand_x_seq
                        new_u_seq = cand_u_seq
                        new_cost = cand_cost
                        u_seq = cand_u_seq
                        break

                if self.enable_log:
                    self.logs.append({
                        'iteration': iteration,
                        'x_seq': x_seq,
                        'u_seq': u_seq,
                        'current_cost': current_cost,
                        'k_seq_seg': k_seq_seg,
                        'K_seq_seg': K_seq_seg,
                        'alpha': chosen_alpha,
                        'new_x_seq': new_x_seq,
                        'new_u_seq': new_u_seq,
                        'new_cost': new_cost,
                        'found_update': found_update
                    })

                self.total_iter = iteration
                if verbose:
                    print(f"Iteration {iteration}, Cost: {new_cost:.4f}, Alpha: {chosen_alpha}")

                if (not found_update) or (abs(current_cost - new_cost) < self.tol):
                    if verbose:
                        print("Convergence achieved or no improvement (with TF).")
                    break

            self.u = u_seq
            final_x_seq = self.simulate(u_seq)
            return u_seq, final_x_seq

    def get_time(self):
        """
        Retrieve the timing logs for total, backward pass, forward pass, and (optionally) inference times.

        Returns:
            tuple: (total_time, backward_pass_time, forward_pass_time[, inference_time])
                   If self.tf is None, no inference_time list is returned.
        """
        if self.tf is not None:
            return (self.total_time, self.backward_pass_time,
                    self.forward_pass_time, self.inference_time)
        else:
            return (self.total_time, self.backward_pass_time, self.forward_pass_time)

    def set_state_offset(self, x_offset):
        self.state_offset = x_offset

    def get_state_offset(self):
        return self.state_offset.copy()