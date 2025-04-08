import mujoco
import numpy as np
import time
import glfw
import pickle
from typing import Optional, List, Dict, Any

# --------------------------------------------------
# Import the new MPC and Transformer modules
# --------------------------------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cartpole_mpc import CartPoleMPC
from quattro_ilqr_tf.transformer_ilqr import *

def run_cartpole_sim(
    model_path: str = "examples/asset/cart_pole/cartpole.xml",
    use_gui: bool = True,
    sweep_positions: Optional[List[float]] = None,
    sweep_angles: Optional[List[float]] = None,
    sim_steps: int = 400,
    horizon: int = 50,
    enable_mpc_sol_log: bool = False,
    mpc_log_filename: str = "ilqr_logs.pkl",
    mpc_transformer_model: Optional[Any] = None
) -> List[Any]:
    """
    Run CartPole simulation(s) using a CartPoleMPC (iLQR-based) controller.
    
    This function logs the actual state/control trajectory and (optionally) the MPC 
    predicted trajectories at each control step. These logs can later be used to train 
    a transformer model to predict the actual system input given the states.
    
    Parameters
    ----------
    model_path : str
        Path to the MuJoCo XML model file.
    use_gui : bool
        If True, opens a GLFW window for real-time visualization.
    sweep_positions : Optional[List[float]]
        List of cart initial positions for simulation runs. If None, defaults to [0.0].
    sweep_angles : Optional[List[float]]
        List of pole initial angles for simulation runs. If None, defaults to [0.8].
    sim_steps : int
        Number of simulation steps for each run.
    horizon : int
        Prediction horizon for the MPC controller.
    enable_mpc_sol_log : bool
        If True, logs the full MPC solution (predicted state and control trajectories) at each step.
    mpc_log_filename : str
        Filename for saving MPC logs.
    mpc_transformer_model : Optional[Any]
        Optional transformer model for the MPC controller.
        
    Returns
    -------
    List[Any]
        A list of MPC logs collected from the simulation runs.
    """
    # --------------------------------------------------
    # 1. Initialize the MuJoCo model and simulation data.
    # --------------------------------------------------
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # --------------------------------------------------
    # 2. Initialize the MPC Controller.
    # --------------------------------------------------
    mpc_controller = CartPoleMPC(
        horizon=horizon,
        dt=0.01,
        integration_method="euler",
        transformer_model=mpc_transformer_model,
        lqr_only=False,
        ilqr_only=False,
        ilqr_tf_only=True,
        ilqr_tf_blend=False
    )
    # Set the tolerance and desired reference state.
    mpc_controller.ilqr.tol = 1e-1
    mpc_controller.x_ref = np.zeros(4)
    
    # --------------------------------------------------
    # 3. Setup GLFW visualization (if enabled).
    # --------------------------------------------------
    window = None
    if use_gui:
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        window = glfw.create_window(800, 600, "CartPole MuJoCo", None, None)
        if window is None:
            glfw.terminate()
            raise Exception("GLFW window creation failed")
        glfw.make_context_current(window)
    
        # Create visualization objects for MuJoCo.
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=8000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    
        # Configure the camera view.
        cam.lookat[:] = [0, 0, 0.5]
        cam.distance = 2.0
        cam.azimuth = 90.0
        cam.elevation = -20.0
    
        # Key callback to adjust the cart reference position.
        def key_callback(window, key, scancode, action, mods):
            if action in (glfw.PRESS, glfw.REPEAT):
                if key == glfw.KEY_LEFT:
                    mpc_controller.x_ref[0] -= 0.05
                elif key == glfw.KEY_RIGHT:
                    mpc_controller.x_ref[0] += 0.05
    
        glfw.set_key_callback(window, key_callback)
    
    # --------------------------------------------------
    # 4. Define initial conditions for the simulation sweep.
    # --------------------------------------------------
    if sweep_positions is None:
        sweep_positions = [0.0]
    if sweep_angles is None:
        sweep_angles = [0.8]
    
    # Retrieve the actuator ID by name ("slide").
    slider_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide")
    
    total_runs = len(sweep_positions) * len(sweep_angles)
    current_run = 0
    
    # --------------------------------------------------
    # 5. Run simulation for each initial condition.
    # --------------------------------------------------
    for init_pos in sweep_positions:
        for init_angle in sweep_angles:
            # Reset simulation data and set the initial conditions.
            mujoco.mj_resetData(model, data)
            data.qpos[0] = init_pos
            data.qpos[1] = init_angle
    
            for step_i in range(sim_steps):
                # Extract the current state:
                # [cart_x, cart_xdot, pole_theta, pole_thetadot]
                cart_x = data.qpos[0]
                pole_theta = data.qpos[1]
                cart_xdot = data.qvel[0]
                pole_thetadot = data.qvel[1]
                x_current = np.array([cart_x, cart_xdot, pole_theta, pole_thetadot])
    
                # Obtain control input from the MPC controller.
                mpc_solution, control_input = mpc_controller.control_step(x_current)
    
                # Apply the control input (note the negative sign).
                data.ctrl[slider_actuator_id] = -control_input[0].item()
    
                # Step the simulation forward.
                mujoco.mj_step(model, data)
    
                # Render the scene if the GUI is enabled.
                if use_gui:
                    if glfw.window_should_close(window):
                        break  # Exit if the window is closed.
                    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    mujoco.mjr_render(mujoco.MjrRect(0, 0, 800, 600), scene, context)
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        mujoco.MjrRect(10, 10, 200, 100),
                        f"Ref X: {mpc_controller.x_ref[0]:.2f}",
                        f"(Left/Right Arrow to adjust)\nActual X: {data.qpos[0]:.2f}, Actual Theta: {data.qpos[1]:.2f}",
                        context,
                    )
                    glfw.swap_buffers(window)
                    glfw.poll_events()
                    time.sleep(0.005)
    
            current_run += 1
            print(f"[{current_run}/{total_runs}] Run finished for init_pos={init_pos}, init_angle={init_angle}.")
    
    # --------------------------------------------------
    # 6. Cleanup and optional logging.
    # --------------------------------------------------
    if use_gui:
        glfw.terminate()
    
    if enable_mpc_sol_log and hasattr(mpc_controller, "ilqr") and mpc_controller.ilqr is not None:
        with open(mpc_log_filename, "wb") as f:
            pickle.dump(mpc_controller.ilqr.logs, f)
        print(f"iLQR solution log saved to '{mpc_log_filename}'.")
    
    # Return collected MPC logs.
    return mpc_controller.ilqr.logs


if __name__ == "__main__":
    # --------------------------------------------------
    # Initialize the Transformer-based iLQR model.
    # --------------------------------------------------
    transformer_model = TransformerILQR(state_dim=4, control_dim=5)
    
    # Load the transformer model weights from a directory.
    model_dir = "examples/cartpole/dec3_dmodel128_nhead4_ff256_drop0.1_epoch200_promptlen5_402.7k"
    transformer_model.load(model_dir)
    
    # --------------------------------------------------
    # Define simulation parameters.
    # --------------------------------------------------
    positions_to_try = [0.0]
    angles_to_try = [0.1]
    
    # Run the CartPole simulation.
    simulation_logs = run_cartpole_sim(
        model_path="examples/asset/cart_pole/cartpole.xml",
        use_gui=True,
        sweep_positions=positions_to_try,
        sweep_angles=angles_to_try,
        sim_steps=5000,  # Adjust as needed (e.g., 2500 steps per run)
        horizon=30,
        enable_mpc_sol_log=False,
        mpc_log_filename="ilqr_logs_rangeXXX.pkl",
        mpc_transformer_model=transformer_model
    )
    
    print(f"Collected data from {len(simulation_logs)} runs.")
    