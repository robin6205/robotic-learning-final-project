import os
import time
import argparse
import torch
import numpy as np
import mujoco
import sys

# Check for mujoco_viewer
try:
    import mujoco_viewer
except ImportError:
    print("Error: mujoco_viewer not found.")
    print("Please install with: pip install mujoco-python-viewer")
    sys.exit(1)

# Import policy model
from gail_airl_ppo.network.policy import StateIndependentPolicy

# Get the absolute path of the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(args):
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    print(f"Loading policy from {args.model_path}")
    
    # Resolve model path
    model_path = args.model_path
    if not model_path.endswith(".pth"):
        model_path = os.path.join(model_path, "actor.pth")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # Load MuJoCo model directly (like in the working script)
    xml_filepath = os.path.join(BASE_DIR, "data", "g1_robot", "g1_23dof_simplified.xml")
    
    try:
        # Load the MuJoCo model and data
        print(f"Loading MuJoCo model from {xml_filepath}")
        model = mujoco.MjModel.from_xml_path(xml_filepath)
        data = mujoco.MjData(model)
        print(f"Successfully loaded model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        sys.exit(1)
    
    # Get state and action dimensions from the model
    state_dim = 46  # 23 joint angles + 23 velocities
    action_dim = model.nu  # Number of actuators
    
    # Initialize policy network
    policy = StateIndependentPolicy(
        state_shape=(state_dim,),
        action_shape=(action_dim,),
        hidden_units=(64, 64),
        hidden_activation=torch.nn.Tanh()
    ).to(device)
    
    # Load policy weights
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    print("Policy loaded successfully")
    
    # For debugging - print a small part of model weights
    for name, param in policy.named_parameters():
        print(f"Parameter {name} shape: {param.shape}")
        if param.numel() > 5:
            print(f"First 5 values: {param.data.flatten()[:5]}")
        break
    
    # Set initial state 
    initial_qpos = np.array([
        0, 0, 0.79,  # Base position (x, y, z)
        1, 0, 0, 0,  # Base orientation (w, x, y, z)
        # Legs (L: P, R, Y, Knee, AP, AR | R: P, R, Y, Knee, AP, AR)
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        # Waist (Y)
        0,
        # Arms (L: SP, SR, SY, Elbow, WR | R: SP, SR, SY, Elbow, WR)
        0.2, 0.2, 0, 1.28, 0,
        0.2, -0.2, 0, 1.28, 0
    ])
    
    # Setup MuJoCo viewer
    print("Setting up MuJoCo viewer...")
    try:
        viewer = mujoco_viewer.MujocoViewer(model, data, title="G1 Robot Policy Visualization")
    except Exception as e:
        print(f"Error setting up viewer: {e}")
        sys.exit(1)
    
    # Run episodes
    for ep in range(1, args.num_episodes + 1):
        print(f"\nStarting episode {ep}")
        
        # Reset simulation state
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)
        
        total_reward = 0.0
        step = 0
        done = False
        
        # For tracking lateral movement and height
        prev_x = data.qpos[0]
        
        while not done and step < args.max_steps and viewer.is_alive:
            try:
                # Get the observation (23 joint angles + 23 velocities)
                joint_angles = data.qpos[-23:].copy()  # Last 23 entries are the joint angles
                joint_vels = data.qvel[-23:].copy()    # Last 23 entries are the joint velocities
                
                # Combine into observation
                obs = np.concatenate([joint_angles, joint_vels])
                
                # Use policy to get action
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action_tensor = policy(obs_tensor)
                action = action_tensor.cpu().numpy().squeeze()
                
                # Apply action to joint actuators
                data.ctrl[:] = action
                
                # Save position for reward calculation
                prev_root_pos = data.qpos[:3].copy()
                
                # Step simulation (run for a few steps to stabilize)
                for _ in range(10):
                    mujoco.mj_step(model, data)
                
                # Calculate reward
                new_root_pos = data.qpos[:3]
                root_height_z = new_root_pos[2]
                
                # Lateral velocity reward
                lateral_velocity_reward = float(new_root_pos[0] - prev_root_pos[0])
                
                # Stability reward
                stability_height_threshold = 0.3
                stability_reward = max(0.0, root_height_z - stability_height_threshold)
                
                # Combine rewards
                stability_reward_weight = 0.1
                reward = lateral_velocity_reward + stability_reward_weight * stability_reward
                
                total_reward += reward
                step += 1
                
                # Check termination
                fall_threshold = 0.2
                done = bool(root_height_z < fall_threshold)
                
                # Render scene
                viewer.render()
                
                # Print step info (every 10 steps)
                if step % 10 == 0:
                    print(f"Step {step}: Reward={reward:.2f}, Height={root_height_z:.2f}, Lateral={lateral_velocity_reward:.2f}")
                
                # Control playback speed
                time.sleep(1.0 / args.fps)
                
            except Exception as e:
                print(f"Error during step {step}: {e}")
                break
        
        print(f"Episode {ep} complete: Reward = {total_reward:.3f}, Steps = {step}")
        
        # Small pause between episodes
        time.sleep(0.5)
    
    # Clean up
    if viewer.is_alive:
        viewer.close()
    
    print("Visualization complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a trained GAIL policy on the G1 robot")
    parser.add_argument('--model_path', type=str, required=True, help='Path to actor.pth or model directory')
    parser.add_argument('--num_episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--fps', type=int, default=30, help='Rendering frame rate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()
    
    try:
        run(args)
    except Exception as e:
        import traceback
        print(f"Error in visualization: {e}")
        traceback.print_exc() 