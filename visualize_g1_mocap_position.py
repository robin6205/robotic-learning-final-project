import mujoco
import mujoco_viewer
import os
import sys
import csv
import numpy as np
import time

# --- Configuration ---
XML_FILENAME = "g1_23dof_simplified.xml"
CSV_FILENAME = "moving_right_position_data.csv"
PLAYBACK_SPEED_FACTOR = 0.25 # Slowed down playback speed
INITIAL_QPOS = np.array([
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
]) # Initial pose (can be None to use XML default)

# --- Helper Functions ---
def get_project_relative_path(*path_parts):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, *path_parts)

def load_mocap_data(csv_filepath):
    """Loads mocap position data and applies rotation to correct orientation."""
    try:
        import pandas as pd
        import re
        
        # Load CSV with pandas
        df = pd.read_csv(csv_filepath)
        if df.empty:
            print("Empty CSV")
            return None, None, None
            
        # Extract timestamps
        timestamps = df.get('Timestamp', pd.Series(range(len(df))))
        timestamps = timestamps/1000.0 if 'Timestamp' in df.columns else timestamps
        
        # Identify joints & build position array
        coord_pat = r'_position_[xyz]$'
        cols = [c for c in df.columns if re.search(coord_pat, c)]
        joints = sorted({re.sub(coord_pat,'', c) for c in cols})
        
        n_frames = len(df)
        n_joints = len(joints)
        
        # Extract raw position data
        raw_pts = np.zeros((n_frames, n_joints, 3))
        for idx, j in enumerate(joints):
            raw_pts[:, idx, 0] = df[f"{j}_position_x"]
            raw_pts[:, idx, 1] = df[f"{j}_position_y"]
            raw_pts[:, idx, 2] = df[f"{j}_position_z"]
            
        # Define rotation matrix - the same one used in stick figure viz
        # Maps: X→X, Y→Z, Z→Y (Z-up world, X-forward, Y-sideways)
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ])
        
        # Apply rotation to data
        rot_pts = raw_pts @ R.T
        
        # Fix ground level
        floor = rot_pts[:, :, 2].max()  # highest raw Z becomes ground
        rot_pts[:, :, 2] -= floor
        
        # Return joint names, rotated positions, and timestamps
        return joints, rot_pts, timestamps.tolist()
        
    except Exception as e:
        print(f"Error processing position data: {e}")
        return None, None, None

def get_actuator_indices(model, csv_header):
    """Creates a mapping from CSV column index to MuJoCo ctrl index."""
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    # print(f"Model Actuators: {actuator_names}")

    # Define mapping from CSV names (potentially different) to actuator names
    # Handle known discrepancies like "Flexion", "Pronation"
    csv_to_actuator_map = {
        'LeftHip_pitch': 'left_hip_pitch_joint',
        'LeftHip_roll': 'left_hip_roll_joint',
        'LeftHip_yaw': 'left_hip_yaw_joint',
        'LeftKnee_flexion': 'left_knee_joint',
        'LeftAnkle_pitch': 'left_ankle_pitch_joint',
        'LeftAnkle_roll': 'left_ankle_roll_joint',
        'RightHip_pitch': 'right_hip_pitch_joint',
        'RightHip_roll': 'right_hip_roll_joint',
        'RightHip_yaw': 'right_hip_yaw_joint',
        'RightKnee_flexion': 'right_knee_joint',
        'RightAnkle_pitch': 'right_ankle_pitch_joint',
        'RightAnkle_roll': 'right_ankle_roll_joint',
        'LeftShoulder_pitch': 'left_shoulder_pitch_joint',
        'LeftShoulder_roll': 'left_shoulder_roll_joint',
        'LeftShoulder_yaw': 'left_shoulder_yaw_joint',
        'LeftElbow_flexion': 'left_elbow_joint',
        'RightShoulder_pitch': 'right_shoulder_pitch_joint',
        'RightShoulder_roll': 'right_shoulder_roll_joint',
        'RightShoulder_yaw': 'right_shoulder_yaw_joint',
        'RightElbow_flexion': 'right_elbow_joint',
        'LeftWrist_pronation': 'left_wrist_roll_joint', # Mapping Pronation to Roll
        'RightWrist_pronation': 'right_wrist_roll_joint',# Mapping Pronation to Roll
        'Waist_yaw': 'waist_yaw_joint'
    }

    # Create the mapping: csv_col_index -> ctrl_index
    index_map = {}
    missing_csv = []
    missing_actuators = list(actuator_names)

    for csv_idx, csv_name in enumerate(csv_header):
        actuator_name = csv_to_actuator_map.get(csv_name)
        if actuator_name:
            try:
                ctrl_idx = actuator_names.index(actuator_name)
                index_map[csv_idx] = ctrl_idx
                if actuator_name in missing_actuators:
                    missing_actuators.remove(actuator_name)
            except ValueError:
                # Actuator name from map not found in model (shouldn't happen if map is correct)
                print(f"Warning: Actuator '{actuator_name}' (mapped from CSV '{csv_name}') not found in model actuators.")
        else:
            missing_csv.append(csv_name)

    if missing_csv:
        print(f"Warning: CSV columns not mapped to actuators: {missing_csv}")
    if missing_actuators:
        print(f"Warning: Model actuators not found in CSV mapping: {missing_actuators}")

    if len(index_map) != model.nu:
         print(f"Warning: Mismatch between mapped CSV columns ({len(index_map)}) and number of actuators ({model.nu}). Check mappings.")

    return index_map

# --- Main Script ---

# Construct file paths
xml_filepath = get_project_relative_path("gail-airl-ppo.pytorch", "data", "g1_robot", XML_FILENAME)
csv_filepath = get_project_relative_path("gail-airl-ppo.pytorch", "data", "mocap", CSV_FILENAME)

# Check dependencies
try:
    import mujoco_viewer
except ImportError:
    print("Error: mujoco_viewer not found.")
    print("Please install the provider package: pip install mujoco-python-viewer")
    sys.exit(1)

# --- Load Model and Data ---
try:
    model = mujoco.MjModel.from_xml_path(xml_filepath)
    data = mujoco.MjData(model)
    print(f"Successfully loaded model: {XML_FILENAME} (nq={model.nq}, nv={model.nv}, nu={model.nu})")
except Exception as e:
    print(f"Error loading MuJoCo model from {xml_filepath}: {e}")
    sys.exit(1)

joint_names, position_data, timestamps = load_mocap_data(csv_filepath)
if joint_names is None or position_data.shape[0] == 0:
    print("Error: No valid mocap data loaded")
    sys.exit(1)

# --- Initialize Simulation State ---
if INITIAL_QPOS is not None:
    if len(INITIAL_QPOS) == model.nq:
        data.qpos[:] = INITIAL_QPOS
    else:
        print(f"Warning: INITIAL_QPOS length ({len(INITIAL_QPOS)}) does not match model.nq ({model.nq}). Using default init.")
        mujoco.mj_resetData(model, data)
else:
    mujoco.mj_resetData(model, data) # Use default state from XML

mujoco.mj_forward(model, data) # Run forward dynamics to settle initial state

# --- Setup Viewer ---
print("Launching MuJoCo viewer...")
try:
    viewer = mujoco_viewer.MujocoViewer(model, data, title="Mocap Position Visualization")
except Exception as e:
    print(f"Error launching viewer: {e}")
    sys.exit(1)

print(f"Viewer launched. Playing Mocap data at {PLAYBACK_SPEED_FACTOR}x speed.")
print("Close the viewer window to exit.")

# --- Simulation Loop (Automatic Playback with position visualization) ---
frame_index = 0
sim_step = 0
last_render_time = time.time()
playback_start_time = time.time() # Wall clock time when playback started

# Create site objects in MuJoCo for visualizing joint positions
site_ids = []
for i, joint_name in enumerate(joint_names):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"site_{joint_name}")
    if site_id == -1:
        # Site doesn't exist, add it programmatically (if possible)
        print(f"Warning: Site for joint {joint_name} not found in model")
    else:
        site_ids.append(site_id)

while viewer.is_alive and frame_index < position_data.shape[0]:
    # Determine target simulation time based on wall clock and speed factor
    elapsed_wall_time = time.time() - playback_start_time
    target_sim_time = elapsed_wall_time * PLAYBACK_SPEED_FACTOR
    
    # Find the latest mocap frame whose timestamp is <= target_sim_time
    while frame_index < len(timestamps) and timestamps[frame_index] <= target_sim_time:
        # Set the robot configuration based on IK or some other method (not implemented)
        # For now, we'll just update any available mocap sites with the position data
        for i, site_id in enumerate(site_ids):
            if i < position_data.shape[1]:  # If we have position data for this site
                data.site_xpos[site_id] = position_data[frame_index, i]
        
        # Step physics simulation
        mujoco.mj_step(model, data)
        sim_step += 1
        
        # Advance to next frame
        frame_index += 1
        if frame_index >= position_data.shape[0]:
            break
    
    # Render the scene
    viewer.render()
    
    # Add a small sleep to prevent 100% CPU usage
    time.sleep(0.001)

print(f"Finished visualizing {frame_index} mocap frames.")

# --- Cleanup ---
if viewer and viewer.is_alive:
    viewer.close() 