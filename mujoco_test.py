import gymnasium as gym
import numpy as np

def main():
    # Create the environment
    env = gym.make('InvertedPendulum-v4', render_mode="human")
    
    # Reset the environment
    observation, info = env.reset()
    
    # Run for 1000 steps
    for _ in range(1000):
        # Take a random action
        action = env.action_space.sample()
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Check if the episode is done
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    main() 