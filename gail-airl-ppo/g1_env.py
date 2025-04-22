import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# Get base directory for consistent file references
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class G1Env(gym.Env):
    """
    Custom Gym environment for the Unitree G1 robot
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(G1Env, self).__init__()
        
        # Load the model
        model_path = os.path.join(BASE_DIR, "data/g1_robot/g1_23dof_simplified.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Get the actual dimensions from the model
        self.qpos_dim = self.model.nq
        self.qvel_dim = self.model.nv
        
        print(f"Loaded model with qpos dimension: {self.qpos_dim}, qvel dimension: {self.qvel_dim}")
        
        # Define action space (joint angle changes)
        # Assuming 23 controllable joints based on the model file name
        n_dof = 23
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(n_dof,), dtype=np.float32
        )
        
        # Define observation space based on our trimmed observation (23 joint angles + 23 velocities)
        # This matches the expert data dimensions
        obs_dim = 46  # 23 joint angles + 23 velocities
        print(f"Using observation dimension: {obs_dim} (trimmed to match expert data)")
        
        high = np.ones(obs_dim, dtype=np.float32) * np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32
        )
        
        # Set up rendering
        self.render_mode = render_mode
        self.viewer = None
        
        # Initialize renderer if human mode is requested
        if self.render_mode == "human":
            self._setup_renderer()
    
    def _setup_renderer(self):
        """Set up the MuJoCo renderer"""
        from mujoco.glfw import glfw
        
        # Initialize glfw if not already initialized
        if not glfw.glfwInit():
            raise RuntimeError("Failed to initialize GLFW")
            
        # Create window
        self.window = glfw.glfwCreateWindow(1200, 900, "G1 Robot", None, None)
        if not self.window:
            glfw.glfwTerminate()
            raise RuntimeError("Failed to create GLFW window")
            
        # Make context current
        glfw.glfwMakeContextCurrent(self.window)
        
        # Create renderer
        self.renderer = mujoco.Renderer(self.model, 1200, 900)
    
    def _get_obs(self):
        """Get the current observation"""
        # Get joint positions (qpos) and velocities (qvel)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # In MJCF, qpos includes:
        # - 3 values for the root position (x, y, z)
        # - 4 values for the root orientation (quaternion)
        # - The rest are joint angles
        # We only want the joint angles (last 23 elements) to match expert data
        if len(qpos) > 23:
            qpos_joints = qpos[-23:]  # Take only the last 23 joint values
        else:
            qpos_joints = qpos  # Keep all if already right size
            
        # Similarly for qvel
        if len(qvel) > 23:
            qvel_joints = qvel[-23:]  # Take only the last 23 velocity values
        else:
            qvel_joints = qvel  # Keep all if already right size
        
        # Concatenate to get the full state and convert to float32
        obs = np.concatenate([qpos_joints, qvel_joints]).astype(np.float32)
        return obs
        
    def reset(self, seed=None, options=None):
        """Reset the environment to a random initial state"""
        super().reset(seed=seed)
        
        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial configuration to a neutral position
        self.data.qpos[:] = 0.0
        
        # Forward dynamics to get the simulation into a valid state
        mujoco.mj_forward(self.model, self.data)
        
        # Get the observation
        obs = self._get_obs()
        
        # Extra info
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Step the simulation forward based on the action"""
        # Apply action (change in joint angles)
        # Clip to ensure within action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # We can only control the actual joint DOFs, not the root position/orientation
        # Skip the first 7 qpos values (3 for position, 4 for quaternion orientation)
        # and apply action to the last 23 joint values (if model has more)
        joint_start_idx = self.qpos_dim - 23  # Calculate where the last 23 joints start
        joint_start_idx = max(7, joint_start_idx)  # Ensure we're at least past the root
        
        # Apply actions to the joints
        n_action = min(len(action), 23)  # We expect 23 actions
        self.data.qpos[joint_start_idx:joint_start_idx+n_action] += action[:n_action]
        
        # Run the simulation for a few steps to stabilize
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Get the new observation
        obs = self._get_obs()
        # Simple reward structure - can be modified according to task
        reward = 0.0
        # Check for termination
        terminated = False
        truncated = False # Check for truncation (e.g., if agent falls over or moves out of bounds)
        
        # Additional info
        info = {}
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            from mujoco.glfw import glfw
            # Update scene and render
            self.renderer.update_scene(self.data)
            image = self.renderer.render()
            # Update window
            glfw.glfwMakeContextCurrent(self.window)
            glfw.glfwPollEvents()
            
            # Create buffer and update window
            glfw.glfwGetWindowSize(self.window)
            
            # Check if window should close
            if glfw.glfwWindowShouldClose(self.window):
                self.close()
                
        return None
        
    def close(self):
        """Clean up resources"""
        if self.viewer:
            from mujoco.glfw import glfw
            glfw.glfwTerminate()
            self.viewer = None

# Gymnasium to Gym compatibility wrapper
class GymCompatibilityWrapper(gym.Wrapper):
    """Wraps a Gymnasium environment to simulate a gym.Env (old API)"""
    
    def __init__(self, env):
        super(GymCompatibilityWrapper, self).__init__(env)
        
    def step(self, action):
        """Convert Gymnasium step to old gym step format"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
        
    def reset(self):
        """Convert Gymnasium reset to old gym reset format"""
        obs, _ = self.env.reset()
        return obs

# Register the environment
from gymnasium.envs.registration import register

# Register the raw environment
register(
    id='G1Raw-v0',
    entry_point='g1_env:G1Env',
    max_episode_steps=1000,
)

# Create a custom make function for compatibility
def make_g1_env():
    """Create a G1 environment wrapped for gym compatibility"""
    raw_env = gym.make('G1Raw-v0')
    return GymCompatibilityWrapper(raw_env)

# For testing the environment directly
if __name__ == "__main__":
    env = make_g1_env()
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Action shape: {action.shape}")
        print(f"Observation shape: {obs.shape}")
        print(f"Reward: {reward}")
        
        if done:
            break
            
    env.close()
    print("Environment test completed.") 