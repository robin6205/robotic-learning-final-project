import mujoco
import mujoco_viewer
import os
import sys
import csv
import numpy as np
import time

# --- Configuration ---
XML_FILENAME = "g1_23dof_simplified.xml"
CSV_FILENAME = "moving_right_processed_data.csv"
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
    """Loads mocap data, converts degrees to radians, handles mapping."""
    mocap_frames = []
    header = []
    timestamps = []
    try:
        with open(csv_filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Read header row
            # print(f"CSV Header: {header}")
            for row in reader:
                if not row: continue # Skip empty rows
                timestamps.append(float(row[0]) / 1000.0) # Convert ms to seconds
                # Convert data (degrees to radians), skipping timestamp
                mocap_frames.append([np.deg2rad(float(x)) for x in row[1:]])
    except FileNotFoundError:
        print(f"Error: Mocap CSV file not found at {csv_filepath}")
        return None, None, None
    except Exception as e:
        print(f"Error reading Mocap CSV file: {e}")
        return None, None, None

    if not header or not mocap_frames:
        print("Error: Mocap CSV file is empty or header is missing.")
        return None, None, None

    # Return header (without Timestamp), frames, and timestamps
    return header[1:], mocap_frames, timestamps

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

csv_header, mocap_frames, timestamps = load_mocap_data(csv_filepath)
if not mocap_frames:
    sys.exit(1)

index_map = get_actuator_indices(model, csv_header)
if not index_map:
    print("Error: Could not create a valid mapping between CSV columns and actuators.")
    sys.exit(1)
if len(index_map) != model.nu:
    print(f"Warning: Mismatch between mapped CSV columns ({len(index_map)}) and number of actuators ({model.nu}). Check mappings carefully.")

# --- Initialize Simulation State ---
if INITIAL_QPOS is not None:
    if len(INITIAL_QPOS) == model.nq:
        data.qpos[:] = INITIAL_QPOS
    else:
        print(f"Warning: INITIAL_QPOS length ({len(INITIAL_QPOS)}) does not match model.nq ({model.nq}). Using default init.")
        mujoco.mj_resetData(model, data)
else:
    mujoco.mj_resetData(model, data) # Use default state from XML

# Set initial control to the first mocap frame
if index_map and mocap_frames:
    first_frame = mocap_frames[0]
    for csv_idx, ctrl_idx in index_map.items():
        if csv_idx < len(first_frame):
            try:
                data.ctrl[ctrl_idx] = first_frame[csv_idx]
            except IndexError:
                 print(f"Error setting initial control: ctrl index {ctrl_idx} out of bounds.")
                 sys.exit(1)
        else:
            print(f"Warning: CSV index {csv_idx} out of bounds for initial frame 0.")

mujoco.mj_forward(model, data) # Run forward dynamics to settle initial state

# --- Setup Viewer ---
print("Launching MuJoCo viewer...")
try:
    viewer = mujoco_viewer.MujocoViewer(model, data, title="Mocap Playback (Slowed)")
except Exception as e:
    print(f"Error launching viewer: {e}")
    sys.exit(1)

print("Viewer launched. Playing Mocap data at {PLAYBACK_SPEED_FACTOR}x speed.")
print("Close the viewer window to exit.")

# --- Simulation Loop (Automatic Playback) ---
frame_index = 0
sim_step = 0
last_render_time = time.time()
playback_start_time = time.time() # Wall clock time when playback started

while viewer.is_alive and frame_index < len(mocap_frames):
    # Determine target simulation time based on wall clock and speed factor
    elapsed_wall_time = time.time() - playback_start_time
    target_sim_time = elapsed_wall_time * PLAYBACK_SPEED_FACTOR

    # Find the latest mocap frame whose timestamp is <= target_sim_time
    # This allows skipping frames if playback is too slow, or waiting if too fast
    while frame_index < len(mocap_frames) and timestamps[frame_index] <= target_sim_time:
        current_frame = mocap_frames[frame_index]
        valid_frame = True
        for csv_idx in index_map.keys():
            if csv_idx >= len(current_frame):
                print(f"Warning: CSV index {csv_idx} out of bounds for frame {frame_index}. Skipping frame.")
                valid_frame = False
                break # Skip this mocap frame
        
        if valid_frame:
            # Set controls for this frame
            for csv_idx, ctrl_idx in index_map.items():
                try:
                    data.ctrl[ctrl_idx] = current_frame[csv_idx]
                except IndexError:
                    print(f"Error setting control: ctrl index {ctrl_idx} out of bounds for frame {frame_index}. Skipping control set.")
                    valid_frame = False # Mark as invalid to avoid stepping with partial controls
                    break
        
        # Only advance simulation if controls were set successfully
        if valid_frame:
             # Step simulation forward until it catches up to the *current* mocap frame's time
             # Note: This might step multiple times per render loop if playback is slow
            while data.time < timestamps[frame_index]: 
                mujoco.mj_step(model, data)
                sim_step += 1
                if data.time >= timestamps[frame_index]:
                    break # Stop stepping once we've reached or passed the frame time
        
        # Always advance frame index after processing it
        frame_index += 1

    # --- Render --- (regardless of whether a new frame was processed, keep rendering)
    viewer.render()
    # Add a small sleep to prevent 100% CPU usage 
    time.sleep(0.001) 


print(f"Finished playing {frame_index} mocap frames.")

# --- Cleanup ---
if viewer and viewer.is_alive:
    viewer.close() 