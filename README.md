# G1 Robot Imitation Learning Project

This project implements imitation learning techniques (GAIL and AIRL) for the Unitree G1 quadruped robot, based on the PPO algorithm. The implementation addresses numerical stability issues in complex robotic control tasks.

## Setup

You can set up the environment using the following steps:

1. Create a Python 3.10 virtual environment:
   ```bash
   python3.10 -m venv mujoco-env-py310-compat
   source mujoco-env-py310-compat/bin/activate
   ```

2. Install required packages:
   ```bash
   pip install numpy==1.24.3 torch gymnasium tqdm tensorboard mujoco
   ```

3. Install MuJoCo:
   ```bash
   mkdir -p ~/.mujoco
   curl -OL https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz
   tar -xf mujoco210-macos-x86_64.tar.gz -C ~/.mujoco/
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
   export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210/
   pip install "gymnasium[mujoco]"
   ```

## Complete Workflow

### 1. Train Expert (SAC)

First, we train an expert policy using Soft Actor-Critic (SAC):

```bash
python train_expert.py --env_id G1-v0 --num_steps 1000000 --eval_interval 10000 --seed 0
```

This will train an expert policy and save checkpoints in the `logs/G1-v0/sac/seed0-TIMESTAMP/` directory.

### 2. Collect Demonstrations

Next, collect expert demonstrations using the trained policy:

```bash
python collect_demo.py \
    --env_id G1-v0 \
    --weight logs/G1-v0/sac/seed0-TIMESTAMP/model/step1000000/actor.pth \
    --buffer_size 50000 --std 0.01 --p_rand 0.0 --seed 0
```

Replace TIMESTAMP with the actual timestamp from your training run. This will create a buffer file in the `buffers/G1-v0/` directory.

### 3. Train Imitation Learning

#### Standard Training

For standard GAIL training:

```bash
python train_imitation.py \
    --algo gail --env_id G1-v0 \
    --buffer buffers/G1-v0/size50000_std0.01_prand0.0.pth \
    --num_steps 1000000 --eval_interval 5000 --rollout_length 2048 --seed 0
```

#### Stable Training (Recommended)

For training with numerical stability improvements:

```bash
python train_imitation_stable.py \
    --algo gail --env_id G1-v0 \
    --buffer buffers/G1-v0/size50000_std0.01_prand0.0.pth \
    --num_steps 1000000 --eval_interval 5000 --rollout_length 2048 \
    --lr 5e-5 --max_grad_norm 0.5 --entropy_coef 0.01 --seed 0
```

The stable training script includes:
- NaN/Inf detection and handling
- Reduced learning rates
- More aggressive gradient clipping
- Better reward scaling
- Environment stability improvements

You can also try AIRL by changing `--algo gail` to `--algo airl`.

### 4. Evaluate the Policy

Evaluate the trained policy:

```bash
python evaluate_policy.py \
    --env_id G1-v0 \
    --weight logs/G1-v0/gail/seed0-TIMESTAMP/model/step1000000/actor.pth \
    --episodes 10
```

### 5. Visualize Results

#### Visualize Expert

```bash
python visualize_expert.py \
    --env_id G1-v0 \
    --weight logs/G1-v0/sac/seed0-TIMESTAMP/model/step1000000/actor.pth
```

#### Visualize Trained Policy

```bash
python visualize_policy.py \
    --env_id G1-v0 \
    --algo gail \
    --weight logs/G1-v0/gail/seed0-TIMESTAMP/model/step1000000/actor.pth
```

#### View Training Metrics

```bash
tensorboard --logdir logs
```

## Troubleshooting Stability Issues

If you encounter numerical stability issues during training (NaN/Inf values):

1. Use the stable training script (`train_imitation_stable.py`)
2. Decrease the learning rate with `--lr 1e-5`
3. Reduce gradient clipping with `--max_grad_norm 0.5`
4. Increase entropy coefficient with `--entropy_coef 0.02`
5. Use smaller batch sizes with `--batch_size 32`

## Project Structure

- `g1_env.py`: Custom G1 robot environment with stability improvements
- `make_buffer.py`: Create expert demonstration buffers
- `train_expert.py`: Train expert policy using SAC
- `collect_demo.py`: Collect demonstrations from trained policy
- `train_imitation.py`: Original imitation learning training script
- `train_imitation_stable.py`: Enhanced training script with stability features
- `visualize_policy.py`: Visualize trained policy
- `visualize_expert.py`: Visualize expert demonstrations
- `evaluate_policy.py`: Evaluate policy performance
- `gail_airl_ppo/`: Core algorithm implementations
  - `algo/ppo.py`: PPO algorithm
  - `algo/gail.py`: GAIL algorithm
  - `algo/airl.py`: AIRL algorithm
  - `algo/sac.py`: SAC algorithm
  - `buffer.py`: Replay and rollout buffers
  - `network.py`: Neural network architectures
  - `trainer.py`: Training utilities

## References

- Ho, J., & Ermon, S. (2016). Generative Adversarial Imitation Learning. NIPS 2016.
- Fu, J., Luo, K., & Levine, S. (2017). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. arXiv:1710.11248.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML 2018.