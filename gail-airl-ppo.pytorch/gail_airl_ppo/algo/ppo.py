import torch
from torch import nn
from torch.optim import Adam
import os
import numpy as np

from .base import Algorithm
from gail_airl_ppo.buffer import RolloutBuffer
from gail_airl_ppo.network import StateIndependentPolicy, StateFunction


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]
    
    # Normalize and clip advantages for stability
    returns = gaes + values
    gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
    gaes = torch.clamp(gaes, -10.0, 10.0)  # Clip to avoid extreme values

    return returns, gaes


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=1e-4,
                 lr_critic=1e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.01,
                 max_grad_norm=1.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        # Check for NaN/Inf in state
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print("Warning: NaN or Inf detected in state. Replacing with zeros.")
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        action, log_pi = self.explore(state)
        
        # Check for NaN/Inf in action
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print("Warning: NaN or Inf detected in action. Using random action.")
            action = np.random.uniform(-1, 1, size=action.shape)
            # Need to recalculate log_pi for the new action
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
                action_tensor = torch.tensor(action, dtype=torch.float, device=self.device)
                log_pi = self.actor.evaluate_log_pi(state_tensor.unsqueeze(0), 
                                                   action_tensor.unsqueeze(0)).item()
            
        next_state, reward, done, info = env.step(action)
        
        # Check for NaN/Inf in next_state
        if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
            print("Warning: NaN or Inf detected in next_state. Replacing with zeros.")
            next_state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip reward for stability
        reward = np.clip(reward, -10.0, 10.0)
        
        mask = False if t == env._max_episode_steps else done

        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()
            # Check for NaN/Inf in reset state
            if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                print("Warning: NaN or Inf detected in reset state. Replacing with zeros.")
                next_state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
            
        # Check all tensors for NaN/Inf
        tensors_to_check = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'log_pis': log_pis,
            'next_states': next_states
        }
        
        for name, tensor in tensors_to_check.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: NaN or Inf detected in {name}. Cleaning data.")
                if name == 'log_pis':
                    tensor = torch.nan_to_num(tensor, nan=-10.0, posinf=0.0, neginf=-10.0)
                else:
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                tensors_to_check[name] = tensor
                
        # Reassign cleaned tensors
        states, actions, rewards, dones, log_pis, next_states = (
            tensors_to_check['states'],
            tensors_to_check['actions'],
            tensors_to_check['rewards'],
            tensors_to_check['dones'],
            tensors_to_check['log_pis'],
            tensors_to_check['next_states']
        )
                
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        prediction = self.critic(states)
        
        # Check for NaN/Inf in predictions
        if torch.isnan(prediction).any() or torch.isinf(prediction).any():
            print("Warning: NaN or Inf detected in critic prediction. Skipping update.")
            return
            
        loss_critic = (prediction - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        
        # Clip gradient values for improved stability
        for param in self.critic.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1.0, 1.0)
                
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        
        # Check for NaN/Inf in log_pis
        if torch.isnan(log_pis).any() or torch.isinf(log_pis).any():
            print("Warning: NaN or Inf detected in log_pis. Skipping update.")
            return
            
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        # Clip ratios to prevent extreme values
        ratios = torch.clamp(ratios, 0.0, 10.0)
        
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        
        # Clip gradient values for improved stability
        for param in self.actor.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1.0, 1.0)
                
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"Saving PPO model to {save_dir}")
        # Save the actor model
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )
        # Save the critic model
        torch.save(
            self.critic.state_dict(),
            os.path.join(save_dir, 'critic.pth')
        )
