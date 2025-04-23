import torch
import os
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPO
from gail_airl_ppo.network import GAILDiscrim


class GAIL(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=50000, mix_buffer=1,
                 batch_size=64, lr_actor=1e-4, lr_critic=1e-4, lr_disc=1e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_ppo=50, epoch_disc=10,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.01, max_grad_norm=1.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions = self.buffer.sample(self.batch_size)[:2]
            # Samples from expert's demonstrations.
            states_exp, actions_exp = \
                self.buffer_exp.sample(self.batch_size)[:2]
            
            # Check for NaN or inf in states and actions before updating
            if (torch.isnan(states).any() or torch.isinf(states).any() or
                torch.isnan(actions).any() or torch.isinf(actions).any() or
                torch.isnan(states_exp).any() or torch.isinf(states_exp).any() or
                torch.isnan(actions_exp).any() or torch.isinf(actions_exp).any()):
                print("Warning: NaN or Inf detected in batch data. Skipping discriminator update.")
                continue
                
            # Update discriminator.
            self.update_disc(states, actions, states_exp, actions_exp, writer)

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()
        
        # Check and clean data
        if torch.isnan(states).any() or torch.isinf(states).any():
            print("Warning: NaN or Inf detected in states. Cleaning data.")
            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
            
        if torch.isnan(actions).any() or torch.isinf(actions).any():
            print("Warning: NaN or Inf detected in actions. Cleaning data.")
            actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            
        if torch.isnan(log_pis).any() or torch.isinf(log_pis).any():
            print("Warning: NaN or Inf detected in log_pis. Cleaning data.")
            log_pis = torch.nan_to_num(log_pis, nan=-10.0, posinf=0.0, neginf=-10.0)
            
        if torch.isnan(next_states).any() or torch.isinf(next_states).any():
            print("Warning: NaN or Inf detected in next_states. Cleaning data.")
            next_states = torch.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, actions)
        
        # Clip rewards to prevent extreme values
        rewards = torch.clamp(rewards, -10.0, 10.0)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(self, states, actions, states_exp, actions_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        # Clip logits to prevent extreme values
        logits_pi = torch.clamp(logits_pi, -10.0, 10.0)
        logits_exp = torch.clamp(logits_exp, -10.0, 10.0)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        
        # Clip the gradient values before norm clipping for additional stability
        for param in self.disc.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1.0, 1.0)
                
        # Apply gradient norm clipping
        nn.utils.clip_grad_norm_(self.disc.parameters(), self.max_grad_norm)
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)

    def save_models(self, save_dir):
        """Save the actor and discriminator models."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print(f"Saving model to {save_dir}")
        # Save the actor model
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )
        # Save the discriminator model
        torch.save(
            self.disc.state_dict(),
            os.path.join(save_dir, 'disc.pth')
        )
