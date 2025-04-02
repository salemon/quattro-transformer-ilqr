import numpy as np

class QuadrotorDynamics:
    """
    A simplified quadrotor dynamical model.

    State x = [
        x,  y,  z,            # position in inertial frame
        vx, vy, vz,           # linear velocity in inertial frame
        phi, theta, psi,      # roll, pitch, yaw (Euler angles)
        p,   q,    r          # body rates around x, y, z axes
    ]

    Control u = [u1, u2, u3, u4], each ui is thrust from rotor i (N).
    """

    def __init__(self,
                 mass=1.0,
                 Ix=0.02,
                 Iy=0.02,
                 Iz=0.04,
                 arm=0.1,
                 gravity=9.81):
        """
        Parameters
        ----------
        mass : float
            Total mass of the quadrotor (kg).
        Ix : float
            Moment of inertia about body x-axis (kg*m^2).
        Iy : float
            Moment of inertia about body y-axis (kg*m^2).
        Iz : float
            Moment of inertia about body z-axis (kg*m^2).
        arm : float
            Arm length from center to each rotor (m).
        gravity : float
            Gravitational acceleration (m/s^2).
        """
        self.m = mass
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.arm = arm
        self.g = gravity

    def continuous_dynamics(self, x, u):
        """
        Compute dx/dt of the 12D state for the quadrotor.

        Parameters
        ----------
        x : np.ndarray, shape (12,)
            State vector.
        u : np.ndarray, shape (4,)
            Control [u1, u2, u3, u4] (each rotor thrust in N).

        Returns
        -------
        dx_dt : np.ndarray, shape (12,)
            Time derivative of the state.
        """
        # Unpack state
        px, py, pz = x[0],  x[1],  x[2]
        vx, vy, vz = x[3],  x[4],  x[5]
        phi, theta, psi = x[6], x[7], x[8]
        p, q, r = x[9], x[10], x[11]

        # Unpack controls (individual motor thrusts)
        u1, u2, u3, u4 = u
        # total thrust
        T = u1 + u2 + u3 + u4

        # === 1) Position Kinematics: dot(px, py, pz) = (vx, vy, vz)
        pdot = np.array([vx, vy, vz])

        # === 2) Velocity Dynamics in the Inertial Frame
        # Body-to-world rotation for the thrust in inertial axes
        # We'll apply thrust along -Z_body, which is "up" in body coords.
        # But we must rotate that by R_b^i (body-to-inertial).
        #
        # For small angles, you might approximate:
        #   a_x = ( T / m ) * (sin(theta))
        #   a_y = -( T / m ) * (sin(phi))
        #   a_z = -g + ( T / m )*cos(phi)*cos(theta)
        #
        # We'll do a more explicit rotation using Euler angles:
        #   R_b^i = R_z(psi) * R_y(theta) * R_x(phi)
        # Then the thrust vector in body coords is [0, 0, -T].
        # Here T>0 is "pulling" the drone upward along -Z_body.
        #
        # For simplicity, let's build the rotation matrix or apply short direct form.
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth,  sth  = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)

        # In many references, the direction of thrust is +Z_body. We'll assume the 
        # thrust acts along -Z_body (which is typical for the "UAV flips" convention).
        # => Force in the inertial frame is T * R_b^i * ( -z_body ).
        #  z_body = [0, 0, 1], so -z_body = [0, 0, -1].
        #  R_b^i * [0, 0, -1]^T = the 3rd column of R_b^i * -1.
        #
        # But let's do it by direct “drone standard”:
        #   a_x = (T/m) * ( cpsi*sth*cphi + spsi*sphi )
        #   a_y = (T/m) * ( spsi*sth*cphi - cpsi*sphi )
        #   a_z = -g + (T/m)*cth*cphi
        #
        # This is a well-known set if you interpret T as upward (along z_body).
        ax = (T / self.m) * (spsi*sphi + cpsi*sth*cphi)
        ay = (T / self.m) * (cpsi*sphi - spsi*sth*cphi)
        az = -self.g + (T / self.m) * (cth*cphi)

        vdot = np.array([ax, ay, az])

        # === 3) Orientation Kinematics: [phi, theta, psi] dot
        # We'll use standard Euler angle rates:
        #   phi_dot   = p + q sin(phi) tan(theta) + r cos(phi) tan(theta)
        #   theta_dot = q cos(phi) - r sin(phi)
        #   psi_dot   = q sin(phi)/cos(theta) + r cos(phi)/cos(theta)
        #
        # Watch for singularities near theta = +- pi/2.
        phi_dot = p + q*sphi*np.tan(theta) + r*cphi*np.tan(theta)
        theta_dot = q*cphi - r*sphi
        psi_dot = (q*sphi + r*cphi) / np.cos(theta)

        # === 4) Angular Dynamics: [p, q, r] dot
        # With a diagonal inertia: Ix, Iy, Iz
        #   p_dot = ((Iy - Iz)/Ix) * q * r  +  (1 / Ix) * tau_phi
        #   q_dot = ((Iz - Ix)/Iy) * p * r  +  (1 / Iy) * tau_theta
        #   r_dot = ((Ix - Iy)/Iz) * p * q  +  (1 / Iz) * tau_psi
        #
        # Next we find tau_phi, tau_theta, tau_psi from the four motor thrusts.
        # For a "plus" configuration:
        #   tau_phi   ~  arm * (u2 + u3 - u1 - u4)
        #   tau_theta ~  arm * (u3 + u4 - u1 - u2)
        #   tau_psi   ~  k * (u1 + u3 - u2 - u4)
        #
        # But let's do a simple case: X-configuration with "standard" signs:
        tau_phi   = self.arm * ((u2 + u3) - (u1 + u4))   # Roll from left/right difference
        tau_theta = self.arm * ((u1 + u2) - (u3 + u4))   # Pitch from front/back difference
        #
        # For yaw torque, often each rotor creates torque ~ +/- c*u_i, with c some constant 
        # depending on rotor direction. For simplicity here, we'll just do:
        k_yaw = 0.01  # Example constant; adjust as necessary
        # Compute yaw torque:
        tau_psi = k_yaw * (u1 - u2 + u3 - u4)
        #
        # If you want a more accurate model, define a "k_yaw" factor and do:
        # tau_psi = k_yaw*(u1 - u2 + u3 - u4).
        # We'll keep k_yaw=0 for simplicity; iLQR still sees some rotational DOFs from phi/theta.

        p_dot = ((self.Iy - self.Iz) / self.Ix) * (q * r) + (tau_phi / self.Ix)
        q_dot = ((self.Iz - self.Ix) / self.Iy) * (p * r) + (tau_theta / self.Iy)
        r_dot = ((self.Ix - self.Iy) / self.Iz) * (p * q) + (tau_psi / self.Iz)

        # Collect derivatives
        dx_dt = np.array([
            pdot[0], pdot[1], pdot[2],
            vdot[0], vdot[1], vdot[2],
            phi_dot, theta_dot, psi_dot,
            p_dot,   q_dot,     r_dot
        ])

        return dx_dt

    def discrete_dynamics(self, x, u, dt, method="euler"):
        """
        Integrate one timestep to get x_{k+1} from x_k, u_k.

        Parameters
        ----------
        x : np.ndarray, shape (12,)
            Current state.
        u : np.ndarray, shape (4,)
            Current control.
        dt : float
            Integration timestep.
        method : str, optional
            {'euler', 'rk4'}, integration method.

        Returns
        -------
        x_next : np.ndarray, shape (12,)
            Next state after one time step.
        """
        if method == "euler":
            f = self.continuous_dynamics(x, u)
            return x + dt * f

        elif method == "rk4":
            k1 = self.continuous_dynamics(x, u)
            k2 = self.continuous_dynamics(x + 0.5 * dt * k1, u)
            k3 = self.continuous_dynamics(x + 0.5 * dt * k2, u)
            k4 = self.continuous_dynamics(x + dt * k3, u)
            return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        else:
            raise ValueError(f"Unknown integration method: {method}")

    def linearized_dynamics(self, x_eq=None, u_eq=None):
        """
        (Optional) Compute linearization A, B around an equilibrium (x_eq, u_eq).
        For example, hover at [0,0,0, 0,0,0, 0,0,0, 0,0,0] with
        u_eq = [T_hover/4, T_hover/4, T_hover/4, T_hover/4].

        Parameters
        ----------
        x_eq : np.ndarray, shape (12,), optional
            Equilibrium state. If None, assume hover at origin with zero angles.
        u_eq : np.ndarray, shape (4,), optional
            Equilibrium control. If None, assume T_hover that cancels gravity.

        Returns
        -------
        A : np.ndarray, shape (12,12)
            The Jacobian df/dx at (x_eq, u_eq).
        B : np.ndarray, shape (12,4)
            The Jacobian df/du at (x_eq, u_eq).
        """
        if x_eq is None:
            # e.g. hover at: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            x_eq = np.zeros(12)
        if u_eq is None:
            # total thrust = mg => T = m*g
            # each motor = m*g/4
            T_hover = self.m * self.g
            u_eq = np.ones(4) * (T_hover / 4.0)

        # We'll do a numerical Jacobian around (x_eq, u_eq).
        # For iLQR, you can use the full nonlinear model; you don't strictly need A, B,
        # but let's provide them for completeness.

        # Dimensions
        n = 12
        m = 4
        A = np.zeros((n, n))
        B = np.zeros((n, m))

        f0 = self.continuous_dynamics(x_eq, u_eq)

        eps = 1e-6

        # Jacobian wrt x
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            f_plus  = self.continuous_dynamics(x_eq + dx, u_eq)
            f_minus = self.continuous_dynamics(x_eq - dx, u_eq)
            A[:, i] = (f_plus - f_minus) / (2.0 * eps)

        # Jacobian wrt u
        for j in range(m):
            du = np.zeros(m)
            du[j] = eps
            f_plus  = self.continuous_dynamics(x_eq, u_eq + du)
            f_minus = self.continuous_dynamics(x_eq, u_eq - du)
            B[:, j] = (f_plus - f_minus) / (2.0 * eps)

        return A, B
