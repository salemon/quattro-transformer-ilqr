import mujoco
import numpy as np
import time
import glfw
import pickle

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from quadrotor_wrapper import QuadrotorWrapper
from quadrotor_mpc import QuadrotorMPC

from quattro_ilqr_tf.transformer_ilqr import *

# --- Helper functions ---

def euler2quat(phi, theta, psi):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z].
    """
    hp, ht, hy = phi/2, theta/2, psi/2
    cp, sp = np.cos(hp), np.sin(hp)
    ct, st = np.cos(ht), np.sin(ht)
    cy, sy = np.cos(hy), np.sin(hy)
    w = cp * ct * cy + sp * st * sy
    x = sp * ct * cy - cp * st * sy
    y = cp * st * cy + sp * ct * sy
    z = cp * ct * sy - sp * st * cy
    return np.array([w, x, y, z])

def quat_to_rpy(w, x, y, z):
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw).
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)
    else:
        pitch = np.arcsin(sinp)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    return roll, pitch, yaw

# --- Main Simulation Function ---

def run_quadrotor_sim(
    model_path="./asset/skydio_x2/scene.xml",
    use_gui=True,
    sweep_inits=None,
    sim_steps=1000,
    horizon=30,
    dt=0.01,
    enable_mpc_sol_log=False,
    mpc_log_filename="quad_mpc_logs.pkl",
    mpc_transformer_model=None  # transformer model remains in signature for future use
):
    """
    Run quadrotor simulation(s) using a QuadrotorMPC (iLQR-based) controller.
    The function logs state/control trajectories and optionally logs full MPC
    predicted trajectories at each control step.

    Parameters
    ----------
    model_path : str
        Path to the MuJoCo XML model file for the quadrotor.
    use_gui : bool
        If True, opens a GLFW window for visualization.
    sweep_inits : list of dict, optional
        Each dict specifies an initial condition, e.g.:
          { "x": 0.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0 }
        If None, defaults to a single run.
    sim_steps : int
        Number of simulation steps per trial.
    horizon : int
        MPC horizon length.
    enable_mpc_sol_log : bool
        If True, logs full MPC solution trajectories.
    mpc_log_filename : str
        Filename to save MPC logs.
    mpc_transformer_model : object, optional
        Optional transformer model for the MPC controller.

    Returns
    -------
    logs : list
        Collected MPC logs.
    """
    # 1) Create the quadrotor wrapper (loads model and dynamics)
    quad = QuadrotorWrapper(model_path)
    
    # 2) Create the Quadrotor MPC controller using the new class.
    mpc_controller = QuadrotorMPC(
        horizon=horizon,
        dt=dt,
        integration_method="euler",  # or "euler", as desired
        transformer_model=mpc_transformer_model,
        log_filename=mpc_log_filename
    )
    # mpc_controller.ilqr.tol = 1e-1  # Set the iLQR solver tolerance
    # Set the reference state (for hover, e.g., at z=0.5)
    mpc_controller.x_ref = np.zeros(12)
    mpc_controller.x_ref[2] = 0.5

    # 3) Optionally initialize GLFW and rendering objects
    window, scene, context, cam = None, None, None, None

    if use_gui:
        if not glfw.init():
            raise Exception("GLFW failed to initialize")
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, False)
        width, height = 1200, 900
        window = glfw.create_window(width, height, "Quadrotor MuJoCo", None, None)
        if not window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        glfw.make_context_current(window)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(quad.model, maxgeom=10000)
        context = mujoco.MjrContext(quad.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        cam.lookat[:] = quad.model.stat.center
        cam.distance = 2.0
        cam.elevation = -20
        cam.azimuth = 150

        def key_callback(window, key, scancode, action, mods):
            if action in [glfw.PRESS, glfw.REPEAT]:
                # Example: adjust desired altitude reference
                if key == glfw.KEY_W:
                    mpc_controller.x_ref[0] += 0.5
                elif key == glfw.KEY_A:
                    mpc_controller.x_ref[1] += 0.5
                elif key == glfw.KEY_S:
                    mpc_controller.x_ref[0] -= 0.5
                elif key == glfw.KEY_D:
                    mpc_controller.x_ref[1] -= 0.5
                elif key == glfw.KEY_Q:
                    mpc_controller.x_ref[8] += 0.5
                elif key == glfw.KEY_E:
                    mpc_controller.x_ref[8] -= 0.5
        glfw.set_key_callback(window, key_callback)

    # 4) Determine initial conditions to sweep.
    # For a quadrotor, each initial condition is a dict with x, y, z, roll, pitch, yaw.
    if sweep_inits is None:
        sweep_inits = [{"x": 0.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}]
    
    total_runs = len(sweep_inits)
    run_count = 0

    # For individual actuator assignment, get actuator IDs for each rotor.
    thrust1_id = mujoco.mj_name2id(quad.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust1")
    thrust2_id = mujoco.mj_name2id(quad.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust2")
    thrust3_id = mujoco.mj_name2id(quad.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust3")
    thrust4_id = mujoco.mj_name2id(quad.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust4")

    for init_cond in sweep_inits:
        run_count += 1
        quad.reset()
        # Set initial positions in qpos.
        quad.data.qpos[0] = init_cond.get("x", 0.0)
        quad.data.qpos[1] = init_cond.get("y", 0.0)
        quad.data.qpos[2] = init_cond.get("z", 0.5)
        phi = init_cond.get("roll", 0.0)
        theta = init_cond.get("pitch", 0.0)
        psi = init_cond.get("yaw", 0.0)
        quat = euler2quat(phi, theta, psi)
        quad.data.qpos[3:7] = quat
        # Set initial velocities to zero.
        quad.data.qvel[:] = 0.0
        
        # Simulation loop for this initial condition.
        i = 0
        for step_i in range(sim_steps):
            # Build the 12D state:
            # Positions: qpos[0:3], orientation: convert qpos[3:7] to Euler angles,
            # linear velocities: qvel[0:3], angular velocities: qvel[3:6]
            x_state = np.zeros(12)
            x_state[0:3] = quad.data.qpos[0:3]
            x_state[3:6] = quad.data.qvel[0:3]

            x_state[1] *= -1  # Flip y-axis
            x_state[4] *= -1  # Flip y-axis

            qw, qx, qy, qz = quad.data.qpos[3:7]
            roll_cur, pitch_cur, yaw_cur = quat_to_rpy(qw, qx, qy, qz)
            # print(f"Step {step_i}: rpy={roll_cur}, {pitch_cur}, {yaw_cur}")
            x_state[6:9] = [roll_cur, pitch_cur, yaw_cur]
            x_state[9:12] = quad.data.qvel[3:6]
            if step_i % 20 == 0:
                i += 1
                # Get control from the MPC controller.
                t_start = time.time()
                x_pred, u_pred = mpc_controller.control_step(x_state)
                t_end = time.time()
                print(f"Step {i}: u_pred={u_pred[0]},time:{(t_end - t_start)*1000:.1f} ms")
                # u_pred should be a 4D vector, one entry per rotor.
                # Assign each actuator individually using the actuator IDs.
                quad.data.ctrl[thrust1_id] = u_pred[0][0].item()
                quad.data.ctrl[thrust2_id] = u_pred[0][1].item()
                quad.data.ctrl[thrust3_id] = u_pred[0][2].item()
                quad.data.ctrl[thrust4_id] = u_pred[0][3].item()
            
            # Step simulation.
            quad.step_simulation()
            
            # Rendering.
            if use_gui:
                if glfw.window_should_close(window):
                    break
                mujoco.mjv_updateScene(quad.model, quad.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjr_render(viewport, scene, context)
                glfw.swap_buffers(window)
                glfw.poll_events()
                time.sleep(0.005)
        print(f"[{run_count}/{total_runs}] Finished run for init: {init_cond}")
    
    # Cleanup.
    if use_gui:
        glfw.terminate()
    
    if enable_mpc_sol_log:
        with open(mpc_log_filename, "wb") as f:
            pickle.dump(mpc_controller.ilqr.logs, f)
        print(f"MPC solution log saved to '{mpc_log_filename}'.")

    return mpc_controller.ilqr.logs

if __name__ == "__main__":
    # Define a model wrapper
    model_wrapper = TransformerILQR(
        state_dim=12, 
        control_dim=52,
    )
    model_wrapper.device = "cpu"

    # Load the model
    
    model_dir = "examples/quadrotor/dec3_dmodel128_nhead4_ff512_drop0.1_epoch200_promptlen1_616.2k"
    model_wrapper.load(model_dir)

    # Example: define a sweep of initial conditions.
    init_conditions = [
        {"x": 0.0, "y": 0.0, "z": 0.5, "roll": 0.1, "pitch": 0.0, "yaw": 0.0},
    ]
    
    logs = run_quadrotor_sim(
        model_path="examples/asset/skydio_x2/scene.xml",
        use_gui=True,
        sweep_inits=init_conditions,
        sim_steps=10000,
        horizon=50,
        dt=0.01,
        enable_mpc_sol_log=False,
        mpc_log_filename="quad_mpc_logs.pkl",
        mpc_transformer_model=model_wrapper,
        # mpc_transformer_model=None
    )
    