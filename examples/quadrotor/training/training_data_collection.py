import multiprocessing
import numpy as np
import pickle
import os
import time
import glfw
import mujoco
from scipy.stats import qmc

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quadrotor_wrapper import QuadrotorWrapper
from quadrotor_mpc import QuadrotorMPC

# from quattro_ilqr_tf.transformer_ilqr import *

# --- Helper functions ---
def euler2quat(phi, theta, psi):
    # [Same as your original function]
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
    # [Same as your original function]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    return roll, pitch, yaw

def generate_initial_conditions_lhs(n_samples):
    # [Same as your original function]
    lower_bounds = np.array([-0.5, -0.5, 0.49, -0.2, -0.2, -0.5])
    upper_bounds = np.array([ 0.5,  0.5, 0.51,  0.2,  0.2,  0.5])
    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(sample, lower_bounds, upper_bounds)
    init_conditions = []
    for s in scaled_samples:
        init_conditions.append({
            "x": s[0],
            "y": s[1],
            "z": s[2],
            "roll": s[3],
            "pitch": s[4],
            "yaw": s[5]
        })
    print("Generated {} initial conditions.".format(len(init_conditions)), flush=True)
    return init_conditions

# --- Main Simulation Function for Quadrotor ---
def run_quadrotor_sim(
    model_path="./asset/skydio_x2/scene.xml",
    use_gui=True,
    sweep_inits=None,
    sim_steps=150,
    horizon=50,
    dt=0.01,
    enable_mpc_sol_log=True,
    mpc_log_filename="quad_mpc_logs.pkl",
    flush_interval=10,  # New parameter for periodic flushing
    mpc_transformer_model=None
):
    """
    Run quadrotor simulations. Periodically flushes logs to file and clears in-memory logs.
    """
    print(f"Process {os.getpid()}: Starting simulation with {len(sweep_inits)} initial conditions.", flush=True)
    # 1) Create the quadrotor wrapper.
    quad = QuadrotorWrapper(model_path)
    
    # 2) Create the MPC controller.
    mpc_controller = QuadrotorMPC(
        horizon=horizon,
        dt=dt,
        integration_method="euler",
        transformer_model=mpc_transformer_model,
        log_filename=mpc_log_filename  # This file will be appended to.
    )
    mpc_controller.x_ref = np.zeros(12)
    mpc_controller.x_ref[2] = 0.5

    # 3) Initialize GUI if desired.
    window, scene, context, cam = None, None, None, None
    if use_gui:
        print(f"Process {os.getpid()}: Initializing GUI.", flush=True)
        if not glfw.init():
            raise Exception("GLFW failed to initialize")
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
                if key == glfw.KEY_W:
                    mpc_controller.x_ref[0] += 0.5
                elif key == glfw.KEY_S:
                    mpc_controller.x_ref[0] -= 0.5
                elif key == glfw.KEY_A:
                    mpc_controller.x_ref[1] += 0.5
                elif key == glfw.KEY_D:
                    mpc_controller.x_ref[1] -= 0.5
                elif key == glfw.KEY_Q:
                    mpc_controller.x_ref[8] += 0.5
                elif key == glfw.KEY_E:
                    mpc_controller.x_ref[8] -= 0.5
        glfw.set_key_callback(window, key_callback)

    # 4) Generate initial conditions if none provided.
    if sweep_inits is None:
        sweep_inits = generate_initial_conditions_lhs(100)
    
    total_runs = len(sweep_inits)
    run_count = 0

    # Get actuator IDs.
    thrust1_id = mujoco.mj_name2id(quad.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust1")
    thrust2_id = mujoco.mj_name2id(quad.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust2")
    thrust3_id = mujoco.mj_name2id(quad.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust3")
    thrust4_id = mujoco.mj_name2id(quad.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thrust4")

    for init_cond in sweep_inits:
        run_count += 1
        print(f"Process {os.getpid()}: Starting simulation run {run_count}/{total_runs} for init: {init_cond}", flush=True)
        quad.reset()
        # Set initial state.
        quad.data.qpos[0] = init_cond.get("x", 0.0)
        quad.data.qpos[1] = init_cond.get("y", 0.0)
        quad.data.qpos[2] = init_cond.get("z", 0.5)
        phi = init_cond.get("roll", 0.0)
        theta = init_cond.get("pitch", 0.0)
        psi = init_cond.get("yaw", 0.0)
        quat = euler2quat(phi, theta, psi)
        quad.data.qpos[3:7] = quat
        quad.data.qvel[:] = 0.0
        
        # Simulation loop.
        i = 0
        for step_i in range(sim_steps):
            # Print progress every 500 steps to monitor simulation progress
            if step_i % 500 == 0:
                print(f"Process {os.getpid()}: Simulation run {run_count}/{total_runs} at step {step_i}/{sim_steps}", flush=True)
            x_state = np.zeros(12)
            x_state[0:3] = quad.data.qpos[0:3]
            x_state[3:6] = quad.data.qvel[0:3]
            qw, qx, qy, qz = quad.data.qpos[3:7]
            roll_cur, pitch_cur, yaw_cur = quat_to_rpy(qw, qx, qy, qz)
            x_state[6:9] = [roll_cur, pitch_cur, yaw_cur]
            x_state[9:12] = quad.data.qvel[3:6]
            x_state[1] *= -1
            x_state[4] *= -1
            
            # Get control input.
            if step_i % 20 == 0:
                i += 1
                # Get control from the MPC controller.
                x_pred, u_pred = mpc_controller.control_step(x_state)
                # u_pred should be a 4D vector, one entry per rotor.
                quad.data.ctrl[thrust1_id] = u_pred[0][0].item()
                quad.data.ctrl[thrust2_id] = u_pred[0][1].item()
                quad.data.ctrl[thrust3_id] = u_pred[0][2].item()
                quad.data.ctrl[thrust4_id] = u_pred[0][3].item()
            
            quad.step_simulation()
            
            if use_gui:
                if glfw.window_should_close(window):
                    print(f"Process {os.getpid()}: GLFW window closed; breaking simulation loop.", flush=True)
                    break
                mujoco.mjv_updateScene(quad.model, quad.data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjr_render(viewport, scene, context)
                glfw.swap_buffers(window)
                glfw.poll_events()
                time.sleep(0.005)
        print(f"Process {os.getpid()}: Completed simulation run {run_count}/{total_runs} for init: {init_cond}", flush=True)

        # Flush logs every flush_interval runs.
        if enable_mpc_sol_log and (run_count % flush_interval == 0):
            print(f"Process {os.getpid()}: Flushing logs after {run_count} runs.", flush=True)
            with open(mpc_log_filename, "ab") as f:
                pickle.dump(mpc_controller.ilqr.logs, f)
            # Clear logs from memory.
            mpc_controller.ilqr.logs = []

    if use_gui:
        glfw.terminate()
        print(f"Process {os.getpid()}: Terminated GLFW.", flush=True)
    
    # Flush any remaining logs.
    if enable_mpc_sol_log and mpc_controller.ilqr.logs:
        print(f"Process {os.getpid()}: Flushing final logs.", flush=True)
        with open(mpc_log_filename, "ab") as f:
            pickle.dump(mpc_controller.ilqr.logs, f)
        mpc_controller.ilqr.logs = []
        print(f"Process {os.getpid()}: Final logs flushed to '{mpc_log_filename}'.", flush=True)
    
    # Instead of returning huge logs, return the filename.
    return mpc_log_filename

# --- Multiprocessing Task Wrapper ---
def run_task(init_conditions, flush_interval, log_filename):
    """
    Run simulations for given initial conditions. Returns the log filename.
    """
    print(f"Process {os.getpid()}: Running simulation for {len(init_conditions)} initial conditions.", flush=True)
    log_file = run_quadrotor_sim(
        model_path="examples/asset/skydio_x2/scene.xml",
        use_gui=False,
        sweep_inits=init_conditions,
        sim_steps=2000,
        horizon=50,
        dt=0.01,
        enable_mpc_sol_log=True,
        mpc_log_filename=log_filename,
        flush_interval=flush_interval,
        mpc_transformer_model=None
    )
    print(f"Process {os.getpid()}: Finished simulation task. Log saved to {log_file}", flush=True)
    return log_file

# --- Function to Combine Logs Sequentially ---
def combine_logs_sequentially(group_log_files, combined_log_filename):
    """
    Combine logs from each group file into one combined file.
    Logs are read and appended one-by-one to minimize memory usage.
    """
    print("Starting log combination...", flush=True)
    with open(combined_log_filename, "wb") as fout:
        for log_file in group_log_files:
            print(f"Combining logs from {log_file}...", flush=True)
            with open(log_file, "rb") as fin:
                # Each flush was stored as a separate pickle dump.
                while True:
                    try:
                        logs = pickle.load(fin)
                        for log in logs:
                            pickle.dump(log, fout)
                    except EOFError:
                        break
            # Optionally remove the individual log file.
            os.remove(log_file)
            print(f"Removed individual log file: {log_file}", flush=True)
    print(f"Combined logs saved to '{combined_log_filename}'.", flush=True)

if __name__ == "__main__":
    # Generate 2000 initial conditions using LHS.
    n_samples = 2
    lower_bounds = np.array([-0.3, -0.3, 0.49, -0.2, -0.2, -0.5])
    upper_bounds = np.array([ 0.3,  0.3, 0.51,  0.2,  0.2,  0.5])
    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(sample, lower_bounds, upper_bounds)
    init_conditions = []
    for s in scaled_samples:
        init_conditions.append({
            "x": s[0],
            "y": s[1],
            "z": s[2],
            "roll": s[3],
            "pitch": s[4],
            "yaw": s[5]
        })
    total_inits = len(init_conditions)
    print(f"Total initial condition combinations (via LHS): {total_inits}", flush=True)
    
    # Split the initial conditions into groups.
    num_groups = 10
    print(f"Splitting initial conditions into {num_groups} groups.", flush=True)
    groups = np.array_split(init_conditions, num_groups)
    
    flush_interval = 10  # Flush logs every 10 runs.
    
    print("Starting multiprocessing pool...", flush=True)
    pool = multiprocessing.Pool(processes=num_groups)
    # Each task now returns its log filename.
    tasks = [(group.tolist(), flush_interval, f"quad_mpc_logs_group_{i}.pkl")
         for i, group in enumerate(groups) if len(group) > 0]

    group_log_files = pool.starmap(run_task, tasks)
    pool.close()
    pool.join()
    print("Multiprocessing pool finished.", flush=True)
    
    # Combine group log files sequentially.
    combined_log_filename = "combined_quad_mpc_logs_horizon50.pkl"
    combine_logs_sequentially(group_log_files, combined_log_filename)
    print("All tasks completed.", flush=True)
