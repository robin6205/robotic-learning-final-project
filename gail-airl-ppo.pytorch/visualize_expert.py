import os
import argparse
import torch
import numpy as np
import time

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert

def run(args):
    # Create environment with human rendering mode
    env = make_env(args.env_id, render_mode="human")
    
    # Create expert algorithm
    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weight
    )
    
    print(f"Expert model loaded from {args.weight}")
    
    # Run a few episodes for visualization
    for episode in range(args.num_episodes):
        state = env.reset(seed=args.seed + episode)
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            action = algo.exploit(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Add a small delay to observe the movement better
            time.sleep(0.01)
            
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {episode_steps}")
    
    env.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='InvertedPendulum-v2')
    p.add_argument('--weight', type=str, required=True)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_episodes', type=int, default=5)
    args = p.parse_args()
    run(args) 