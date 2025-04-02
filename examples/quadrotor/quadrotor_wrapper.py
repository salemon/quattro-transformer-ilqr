import mujoco
import numpy as np
import quadrotor_dynamics

class QuadrotorWrapper:
    """
    Minimal wrapper around the MuJoCo Skydio X2 quadrotor model.
    Loads the XML, extracts mass/inertia, and provides methods
    for reading and writing state/controls.
    """

    def __init__(self, xml_file="./asset/skydio_x2/scene.xml"):
        """
        Parameters
        ----------
        xml_file : str
            Path to the MuJoCo model file.
        """
        # 1) Load model & data
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # 2) Find the "x2" body ID
        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "x2"
        )

        # 3) Read total mass & inertia (as compiled by MuJoCo)
        self.mass = self.model.body_mass[self.body_id]               # scalar
        self.inertia_diag = self.model.body_inertia[self.body_id]    # [Ixx, Iyy, Izz]

        # 4) Sensor addresses if you need them
        self.gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro")
        self.acc_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc")
        self.quat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "body_quat")

        self.gyro_adr = self.model.sensor_adr[self.gyro_id]  # 3 values
        self.acc_adr  = self.model.sensor_adr[self.acc_id]   # 3 values
        self.quat_adr = self.model.sensor_adr[self.quat_id]  # 4 values

        # 5) Info about the free joint for the "x2" body
        #    qpos = [x, y, z, qw, qx, qy, qz]
        #    qvel = [vx, vy, vz, wx, wy, wz]
        self.FREE_JOINT_X  = 0
        self.FREE_JOINT_Y  = 1
        self.FREE_JOINT_Z  = 2
        self.FREE_JOINT_QW = 3
        self.FREE_JOINT_QX = 4
        self.FREE_JOINT_QY = 5
        self.FREE_JOINT_QZ = 6

        # Look up the body ID for "x2"
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "x2")

        # Total mass of body "x2"
        self.mass = self.model.body_mass[self.body_id]

        # Diagonal inertia [Ixx, Iyy, Izz] about the body frame
        self.inertia_diag = self.model.body_inertia[self.body_id]

        # 4) Estimate "arm" length from rotor geometry positions
        #    We assume geoms named rotor1..rotor4 are each a single prop at some (x,y).
        rotor_names = ["rotor1", "rotor2", "rotor3", "rotor4"]
        distances = []
        for rname in rotor_names:
            g_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, rname)
            # Position of that geom in the body frame (since it's a child of x2)
            pos = self.model.geom_pos[g_id]
            # Horizontal distance from center
            dist_xy = np.sqrt(pos[0]**2 + pos[1]**2)
            distances.append(dist_xy)

        self.arm = float(np.mean(distances))  # average distance

        # Create a dynamic model instance from the provided QuadrotorDynamics class
        self.dynamic_model = quadrotor_dynamics.QuadrotorDynamics(
            mass=self.mass,
            Ix=self.inertia_diag[0],
            Iy=self.inertia_diag[1],
            Iz=self.inertia_diag[2],
            arm=self.arm,
            gravity=9.81
        )
        # print("Quadrotor initialized with mass:", self.mass, 
        #       "inertia:", self.inertia_diag, "arm:", self.arm)

    def reset(self):
        """
        Reset data to the model's default state (qpos0, qvel0).
        Or you can manually set qpos, qvel, etc. to desired initial conditions.
        """
        mujoco.mj_resetData(self.model, self.data)

    def step_simulation(self):
        """
        Step the MuJoCo simulation forward by one timestep.
        """
        mujoco.mj_step(self.model, self.data)

    def set_controls(self, motor_values):
        """
        Write four motor commands into data.ctrl.
        For example: [M1, M2, M3, M4].
        """
        self.data.ctrl[:] = np.array(motor_values).flatten()

    def get_state(self):
        """
        Return a dictionary or NumPy array of the current state
        (position, velocity, orientation, etc.).
        This is just one possible format.
        """
        state_dict = {}

        # Free joint positions
        qpos = self.data.qpos[:7]   # x, y, z, qw, qx, qy, qz
        qvel = self.data.qvel[:6]   # vx, vy, vz, wx, wy, wz
        state_dict["qpos"] = qpos.copy()
        state_dict["qvel"] = qvel.copy()

        # Sensors
        gyro_data = self.data.sensordata[self.gyro_adr : self.gyro_adr + 3]
        acc_data  = self.data.sensordata[self.acc_adr  : self.acc_adr + 3]
        quat_data = self.data.sensordata[self.quat_adr : self.quat_adr + 4]

        state_dict["gyro"] = gyro_data.copy()
        state_dict["acc"]  = acc_data.copy()
        state_dict["body_quat"] = quat_data.copy()

        return state_dict

    def continuous_dynamics(self, x, u):
        """
        Compute the continuous-time state derivative dx/dt for the quadrotor.
        This method delegates the computation to the underlying dynamic model.

        Parameters
        ----------
        x : np.ndarray, shape (12,)
            The current state.
        u : np.ndarray, shape (4,)
            The control input (motor thrusts).

        Returns
        -------
        dx_dt : np.ndarray, shape (12,)
            The time derivative of the state.
        """
        return self.dynamic_model.continuous_dynamics(x, u)

    def discrete_dynamics(self, x, u, dt, method="euler"):
        """
        Compute the discrete-time next state x_{k+1} given current state x, control u, and timestep dt.
        This method uses the underlying dynamic model to perform numerical integration.

        Parameters
        ----------
        x : np.ndarray, shape (12,)
            The current state.
        u : np.ndarray, shape (4,)
            The control input (motor thrusts).
        dt : float
            The integration timestep.
        method : str, optional
            Integration method, one of {'euler', 'rk4'}. Default is "euler".

        Returns
        -------
        x_next : np.ndarray, shape (12,)
            The state after one time step.
        """
        return self.dynamic_model.discrete_dynamics(x, u, dt, method=method)
