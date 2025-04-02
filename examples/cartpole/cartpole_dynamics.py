import numpy as np

class CartPoleDynamics:
    """
    A class for cart-pole dynamics (theta=0 => upright).
    
    State vector x = [ cart_position, cart_velocity, pole_angle, pole_angular_velocity ]
       - cart_position > 0 means the cart is to the right of the origin
       - pole_angle = 0 means the pole is perfectly upright
         (small positive angle => pole tilts to the right)
    Control u = [ force ],   force > 0 pushes the cart to the right
    """

    def __init__(self, m_cart=1.0, m_pole=0.1, length=0.15, gravity=9.81):
        """
        Parameters
        ----------
        m_cart : float
            Mass of the cart.
        m_pole : float
            Mass of the pole.
        length : float
            Half-length of the pole (distance from pivot to tip).
        gravity : float
            Gravitational acceleration.
        """
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.length = length
        self.gravity = gravity

    def continuous_dynamics(self, x, u):
        """
        Compute the continuous-time state derivative dx/dt for the cart-pole.
        
        Parameters
        ----------
        x : np.ndarray, shape (4,)
            State vector [X, Xdot, theta, theta_dot].
        u : np.ndarray, shape (1,)
            Control input [force].
        
        Returns
        -------
        dx_dt : np.ndarray, shape (4,)
            Time derivative of the state.
        """
        X, Xdot, theta, theta_dot = x
        force = u[0]

        M = self.m_cart
        m = self.m_pole
        l = self.length
        g = self.gravity

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        total_mass = M + m

        # Common term
        temp = (force + m * l * (theta_dot**2) * sin_th) / total_mass

        # Angular acceleration:  (note the sign on gravity with theta=0 at upright)
        theta_ddot = (-g * sin_th + cos_th * temp) / (
            l * (4.0 / 3.0 - (m * cos_th**2) / total_mass)
        )

        # Cart acceleration
        x_ddot = temp - (m * l * theta_ddot * cos_th) / total_mass

        return np.array([Xdot, x_ddot, theta_dot, theta_ddot])

    def discrete_dynamics(self, x, u, dt, method="euler"):
        """
        Compute the discrete-time next state x_{k+1} from x_k, u_k, using either
        Euler or RK4 integration of the continuous dynamics.
        
        Parameters
        ----------
        x : np.ndarray, shape (4,)
            Current state.
        u : np.ndarray, shape (1,)
            Current control.
        dt : float
            Integration timestep.
        method : str, optional
            Integration method, one of {'euler', 'rk4'}, by default "euler"
        
        Returns
        -------
        x_next : np.ndarray, shape (4,)
            Next state after one time step.
        """
        if method == "euler":
            # Forward Euler: x_{k+1} = x_k + dt * f(x_k, u_k)
            f = self.continuous_dynamics(x, u)
            return x + dt * f

        elif method == "rk4":
            # Runge-Kutta 4
            k1 = self.continuous_dynamics(x, u)
            k2 = self.continuous_dynamics(x + 0.5 * dt * k1, u)
            k3 = self.continuous_dynamics(x + 0.5 * dt * k2, u)
            k4 = self.continuous_dynamics(x + dt * k3, u)
            return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        else:
            raise ValueError(f"Unknown integration method: {method}")

    def linearized_dynamics(self):
        """
        Linearizes the cart-pole dynamics around the upright equilibrium point
        (x = [0, 0, 0, 0], u = [0]). 
        
        Returns
        -------
        A : np.ndarray, shape (4, 4)
            The state matrix.
        B : np.ndarray, shape (4, 1)
            The control matrix.
        """
        M = self.m_cart
        m = self.m_pole
        l = self.length
        g = self.gravity
        total_mass = M + m

        # Linearized dynamics matrices (obtained via standard methods)
        A = np.array([
            [0,     1,               0,                   0],
            [0,     0,      - (m * g) / M,                   0],
            [0,     0,               0,                   1],
            [0,     0, ((M + m) * g) / (M * l),              0]
        ])

        B = np.array([
            [0],
            [1 / M],
            [0],
            [-1 / (M * l)]
        ])

        return A, B
