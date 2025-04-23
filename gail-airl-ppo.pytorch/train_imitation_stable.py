import os
import argparse
from datetime import datetime
import torch
import random
import numpy as np

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create the environment
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)
    
    # Load expert buffer
    device = torch.device("cuda" if args.cuda else "cpu")
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=device
    )
    
    # Print debugging information
    print(f"Environment observation space shape: {env.observation_space.shape}")
    print(f"Environment action space shape: {env.action_space.shape}")
    print(f"Expert buffer state shape: {buffer_exp.states.shape}")
    print(f"Expert buffer action shape: {buffer_exp.actions.shape}")
    
    # Check for and fix any NaN or Inf values in the expert buffer
    def check_and_fix_tensor(tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: Found NaN or Inf in expert buffer {name}. Replacing with zeros.")
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor
    
    buffer_exp.states = check_and_fix_tensor(buffer_exp.states, "states")
    buffer_exp.actions = check_and_fix_tensor(buffer_exp.actions, "actions")
    buffer_exp.rewards = check_and_fix_tensor(buffer_exp.rewards, "rewards")
    buffer_exp.dones = check_and_fix_tensor(buffer_exp.dones, "dones")
    buffer_exp.next_states = check_and_fix_tensor(buffer_exp.next_states, "next_states")
    
    # Create algorithm with modified hyperparameters for stability
    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        seed=args.seed,
        rollout_length=args.rollout_length,
        # Specific stability improvements
        lr_actor=args.lr,
        lr_critic=args.lr,
        lr_disc=args.lr,
        epoch_ppo=args.epoch_ppo,
        max_grad_norm=args.max_grad_norm,
        coef_ent=args.entropy_coef,
        # Keep other parameters
        gamma=args.gamma,
        lambd=args.lambd,
        clip_eps=args.clip_eps,
        batch_size=args.batch_size
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}'
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    
    print(f"Starting training with algorithm: {args.algo}")
    print(f"Rollout length: {args.rollout_length}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}, Max grad norm: {args.max_grad_norm}")
    print(f"Entropy coefficient: {args.entropy_coef}, Gamma: {args.gamma}")
    print(f"Lambda: {args.lambd}, Clip epsilon: {args.clip_eps}")
    print(f"PPO epochs: {args.epoch_ppo}")
    print(f"Logs will be saved to: {log_dir}")
    
    # Start training
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        # Save model on error to preserve progress
        algo.save_models(os.path.join(log_dir, f'emergency_save'))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True, help='Path to expert buffer')
    p.add_argument('--rollout_length', type=int, default=2048, help='Rollout length')
    p.add_argument('--num_steps', type=int, default=10**6, help='Number of training steps')
    p.add_argument('--eval_interval', type=int, default=5000, help='Evaluation interval')
    p.add_argument('--env_id', type=str, default='Hopper-v3', help='Environment ID')
    p.add_argument('--algo', type=str, default='gail', help='Algorithm (gail or airl)')
    p.add_argument('--cuda', action='store_true', help='Use CUDA')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Additional stability parameters
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size')
    p.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    p.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    p.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    p.add_argument('--lambd', type=float, default=0.97, help='GAE lambda')
    p.add_argument('--clip_eps', type=float, default=0.2, help='PPO clip epsilon')
    p.add_argument('--epoch_ppo', type=int, default=10, help='PPO epochs per update')
    
    args = p.parse_args()
    run(args) 