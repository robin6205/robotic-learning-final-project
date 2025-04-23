import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import numpy as np # Although we plot degrees, numpy might be useful later

# --- Configuration ---
CSV_FILENAME = "moving_right_processed_data.csv"
OUTPUT_HTML = "mocap_angle_visualization.html"

# --- Helper Functions ---
def get_project_relative_path(*path_parts):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, *path_parts)

# --- Main Script ---

# Construct file path
csv_filepath = get_project_relative_path("gail-airl-ppo.pytorch", "data", "mocap", CSV_FILENAME)

# --- Load Data ---
try:
    df = pd.read_csv(csv_filepath)
    print(f"Successfully loaded Mocap data from: {csv_filepath}")
except FileNotFoundError:
    print(f"Error: Mocap CSV file not found at {csv_filepath}")
    sys.exit(1)
except Exception as e:
    print(f"Error reading Mocap CSV file: {e}")
    sys.exit(1)

if df.empty:
    print("Error: Mocap CSV file is empty.")
    sys.exit(1)

# Convert timestamp to seconds
if 'Timestamp' not in df.columns:
    print("Error: 'Timestamp' column not found in CSV.")
    sys.exit(1)
df['Time (s)'] = df['Timestamp'] / 1000.0

# --- Define Joint Groups and Plot ---

# Define which columns belong to which body part
joint_groups = {
    "Left Leg": ['LeftHip_pitch', 'LeftHip_roll', 'LeftHip_yaw', 'LeftKnee_flexion', 'LeftAnkle_pitch', 'LeftAnkle_roll'],
    "Right Leg": ['RightHip_pitch', 'RightHip_roll', 'RightHip_yaw', 'RightKnee_flexion', 'RightAnkle_pitch', 'RightAnkle_roll'],
    "Left Arm": ['LeftShoulder_pitch', 'LeftShoulder_roll', 'LeftShoulder_yaw', 'LeftElbow_flexion', 'LeftWrist_pronation'],
    "Right Arm": ['RightShoulder_pitch', 'RightShoulder_roll', 'RightShoulder_yaw', 'RightElbow_flexion', 'RightWrist_pronation'],
    "Waist": ['Waist_yaw']
}

num_groups = len(joint_groups)
fig = make_subplots(rows=num_groups, cols=1, shared_xaxes=True,
                    subplot_titles=list(joint_groups.keys()))

# Check if plotly is installed
try:
    import plotly
except ImportError:
    print("Error: Plotly library not found.")
    print("Please install it: pip install pandas plotly")
    sys.exit(1)

row_num = 1
for group_name, joint_list in joint_groups.items():
    for joint_name in joint_list:
        if joint_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Time (s)'],
                    y=df[joint_name],
                    mode='lines',
                    name=joint_name # Label for the legend/hover
                ),
                row=row_num, col=1
            )
        else:
            print(f"Warning: Joint column '{joint_name}' not found in CSV for group '{group_name}'.")
    row_num += 1

fig.update_layout(
    title_text="Mocap Joint Angles Over Time (Degrees)",
    height=300 * num_groups, # Adjust height based on number of plots
    hovermode="x unified" # Show all values at a specific time on hover
)
fig.update_xaxes(title_text="Time (s)", row=num_groups, col=1)
fig.update_yaxes(title_text="Angle (Degrees)")

# --- Save and Instruct ---
try:
    output_path = get_project_relative_path(OUTPUT_HTML)
    fig.write_html(output_path)
    print(f"\nInteractive plot saved to: {output_path}")
    print("Please open this HTML file in your web browser to view the plots.")
    print("You can zoom, pan, and hover over the lines to inspect values.")
except Exception as e:
    print(f"Error writing HTML file: {e}")
    sys.exit(1) 