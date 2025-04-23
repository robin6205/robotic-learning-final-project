import os
import sys
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Configuration ---
CSV_FILENAME  = "moving_right_position_data.csv"
OUTPUT_HTML   = "mocap_stick_figure_visualization.html"
SKELETON_CONNECTIONS = [
    ('Pelvis','Chest'), ('Chest','Head'),
    ('Pelvis','LeftUpperLeg'), ('LeftUpperLeg','LeftLowerLeg'), ('LeftLowerLeg','LeftFoot'),
    ('Pelvis','RightUpperLeg'),('RightUpperLeg','RightLowerLeg'),('RightLowerLeg','RightFoot'),
    ('Chest','LeftShoulder'),('LeftShoulder','LeftUpperArm'),
    ('LeftUpperArm','LeftForeArm'),('LeftForeArm','LeftHand'),
    ('Chest','RightShoulder'),('RightShoulder','RightUpperArm'),
    ('RightUpperArm','RightForeArm'),('RightForeArm','RightHand'),
]

def get_path(*parts):
    return os.path.join(os.path.dirname(__file__), *parts)

# — Load CSV —
df = pd.read_csv(get_path("gail-airl-ppo.pytorch","data","mocap",CSV_FILENAME))
if df.empty:
    print("Empty CSV"); sys.exit(1)

# — Identify joints & build raw array —
coord_pat = r'_position_[xyz]$'
cols      = [c for c in df.columns if re.search(coord_pat, c)]
joints    = sorted({re.sub(coord_pat,'',c) for c in cols})
n_frames  = len(df)
n_joints  = len(joints)

raw_pts = np.zeros((n_frames, n_joints, 3))
for idx, j in enumerate(joints):
    raw_pts[:, idx, 0] = df[f"{j}_position_x"]
    raw_pts[:, idx, 1] = df[f"{j}_position_y"]
    raw_pts[:, idx, 2] = df[f"{j}_position_z"]

# — Define rotation matrix —
# Maps: X→X, Y→Z, Z→Y  (Z-up world, X-forward, Y-sideways)
R = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
])

# — Apply rotation, zero-base feet —
rot = raw_pts @ R.T
floor = rot[:, :, 2].max()      # highest raw Z becomes ground
rot[:, :, 2] -= floor

# — Timestamps —
ts = df.get('Timestamp', pd.Series(range(n_frames)))
time_s = ts/1000.0 if 'Timestamp' in df.columns else ts

# — Build frames —
frames = []
for i, t in enumerate(time_s):
    X, Y, Z = rot[i].T
    mk = go.Scatter3d(x=X, y=Y, z=Z, mode='markers',
                      marker=dict(size=4, color='black'))
    lx, ly, lz = [], [], []
    for a, b in SKELETON_CONNECTIONS:
        ia, ib = joints.index(a), joints.index(b)
        xa, ya, za = rot[i, ia]
        xb, yb, zb = rot[i, ib]
        lx += [xa, xb, None]
        ly += [ya, yb, None]
        lz += [za, zb, None]
    sk = go.Scatter3d(x=lx, y=ly, z=lz, mode='lines',
                      line=dict(width=3, color='blue'))
    frames.append(go.Frame(data=[mk, sk], name=f"{t:.2f}"))

# — Axis ranges from transformed data —
buf = 0.2
all_x = rot[:,:,0].ravel()
all_y = rot[:,:,1].ravel()
all_z = rot[:,:,2].ravel()
x_range = [all_x.min()-buf, all_x.max()+buf]
y_range = [all_y.min()-buf, all_y.max()+buf]
z_range = [all_z.min()-buf, all_z.max()+buf]

# — Build figure —
fig = go.Figure(
    data=frames[0].data,
    frames=frames[1:],
    layout=go.Layout(
        title="Mocap Stick Figure",
        scene=dict(
            xaxis=dict(range=x_range, title="X"),
            yaxis=dict(range=y_range, title="Y"),
            zaxis=dict(range=z_range, title="Z"),
            aspectmode='cube'
        ),
        scene_camera=dict(eye=dict(x=1.5,y=0,z=1.5), up=dict(x=0,y=0,z=1)),
        updatemenus=[dict(type="buttons", buttons=[dict(
            label="Play", method="animate",
            args=[None, {"frame":{"duration":50,"redraw":True}, "fromcurrent":True}]
        )])],
        showlegend=False
    )
)

# — Slider —
steps = [
    dict(method='animate', label=f.name,
         args=[[f.name], {"frame":{"duration":0,"redraw":True},"mode":"immediate"}])
    for f in frames
]
fig.update_layout(sliders=[dict(active=0, pad={"b":10,"t":50}, steps=steps,
                                currentvalue={"prefix":"Time: "})])

# — Save —
out = get_path(OUTPUT_HTML)
fig.write_html(out)
print("Saved animation to", out)
