import pandas as pd
import torch
import numpy as np
import os
import argparse

def make_buffer(csv_path, output_path, time_col='Timestamp', exclude_cols=None):
    """
    Reads humanoid motion data from a CSV file, processes it into state-action pairs,
    and saves it as a PyTorch buffer compatible with the imitation learning setup.

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path where the output .pth buffer file will be saved.
        time_col (str): Name of the timestamp column in the CSV (to exclude from state data)
        exclude_cols (list): Additional column names to exclude from the state
    """
    # print(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Get all columns except timestamp and any other specified excluded columns
    exclude_cols = exclude_cols or []
    if time_col:
        exclude_cols.append(time_col)
    
    # All joint angle columns
    joint_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not joint_cols:
        print(f"Error: No joint angle columns found after excluding {exclude_cols}")
        return
    
    print(f"Found {len(joint_cols)} joint angle columns: {joint_cols}")

    # Extract joint angles
    joint_angles = df[joint_cols].values
    
    # Compute "velocities" as difference between consecutive joint angles
    # This is a simple estimation since we don't have actual velocities
    joint_velocities = np.zeros_like(joint_angles)
    joint_velocities[1:] = joint_angles[1:] - joint_angles[:-1]
    
    if joint_angles.shape[0] < 2:
        print("Error: Need at least two timesteps in the CSV to compute actions.")
        return
    
    # State = [joint_angles, joint_velocities]
    states = np.concatenate([joint_angles, joint_velocities], axis=1)
    
    # Action = delta joint_angles = joint_angles[t+1] - joint_angles[t]
    actions = joint_angles[1:] - joint_angles[:-1]
    
    # Align states with actions: state[t] corresponds to action[t]
    # state[t] -> action[t] -> state[t+1]
    states_t = states[:-1]
    next_states_t = states[1:]
    
    num_transitions = states_t.shape[0]
    print(f"Generated {num_transitions} transitions.")
    
    # Create dummy rewards and dones
    rewards_t = np.zeros((num_transitions, 1), dtype=np.float32)
    # Assume it's one long episode, only the last transition leads to a 'done' state
    dones_t = np.zeros((num_transitions, 1), dtype=np.bool_)
    dones_t[-1] = True
    
    # Convert to PyTorch tensors
    states_tensor = torch.tensor(states_t, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    rewards_tensor = torch.tensor(rewards_t, dtype=torch.float32)
    dones_tensor = torch.tensor(dones_t, dtype=torch.bool)
    next_states_tensor = torch.tensor(next_states_t, dtype=torch.float32)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Save the buffer
    buffer_data = {
        'state': states_tensor,
        'action': actions_tensor,
        'reward': rewards_tensor,
        'done': dones_tensor,
        'next_state': next_states_tensor
    }
    
    try:
        torch.save(buffer_data, output_path)
        print(f"Expert buffer saved successfully to {output_path}")
        print("Buffer content shapes:")
        print(f"  States:      {states_tensor.shape}")
        print(f"  Actions:     {actions_tensor.shape}")
        print(f"  Rewards:     {rewards_tensor.shape}")
        print(f"  Dones:       {dones_tensor.shape}")
        print(f"  Next States: {next_states_tensor.shape}")
    
    except Exception as e:
        print(f"Error saving buffer to {output_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate expert buffer from CSV motion data.")
    parser.add_argument('--csv', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--out', type=str, default='buffers/side_step_expert.pth', help='Path to save the output buffer file.')
    parser.add_argument('--time_col', type=str, default='Timestamp', help='Name of timestamp column to exclude')
    parser.add_argument('--exclude', nargs='+', default=[], help='Additional columns to exclude')
    args = parser.parse_args()

    make_buffer(args.csv, args.out, args.time_col, args.exclude)

    # # Verification step hint
    # print("\nTo verify, run:")
    # # Use single quotes for the inner command string
    # print(f"python -c \"import torch; expert = torch.load('{args.out}'); print(expert['state'].shape, expert['action'].shape)\"") 