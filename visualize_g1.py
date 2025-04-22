import mujoco
import mujoco_viewer
import os
import sys

# Construct the full path to the XML file
xml_filename = "g1_23dof_simplified.xml"
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path relative to the script's location
xml_filepath = os.path.join(script_dir, "gail-airl-ppo.pytorch", "data", "g1_robot", xml_filename)
# Get the directory containing the XML file, which is where MuJoCo expects the 'assets' folder
xml_dir = os.path.dirname(xml_filepath)


# Check if mujoco_viewer is installed (provided by mujoco-python-viewer package)
try:
    import mujoco_viewer
except ImportError:
    print("Error: mujoco_viewer not found.")
    print("Please install the provider package: pip install mujoco-python-viewer")
    sys.exit(1)

try:
    # Load the model
    # MuJoCo automatically looks for meshdir ('assets' in this case) relative to the XML file's directory
    model = mujoco.MjModel.from_xml_path(xml_filepath)
    data = mujoco.MjData(model)

    print(f"Successfully loaded model: {xml_filename}")
    print(f"  Number of bodies: {model.nbody}")
    print(f"  Number of joints (njnt): {model.njnt}")
    print(f"  Number of DoFs (nv): {model.nv}") # Degrees of freedom in velocity space
    print(f"  Number of generalized coordinates (nq): {model.nq}") # Degrees of freedom in position space (includes freejoint)
    print(f"  Number of geoms (ngeom): {model.ngeom}")
    print(f"  Number of sites (nsite): {model.nsite}")
    print(f"  Number of actuators (nu): {model.nu}")
    # The line below caused an AttributeError in newer mujoco versions
    # print(f"Assets are expected in: {os.path.join(xml_dir, model.compiler.meshdir.decode())}")


    # Launch the viewer
    print("Launching MuJoCo viewer...")
    viewer = mujoco_viewer.MujocoViewer(model, data)
    print("Viewer launched. Close the viewer window to exit the script.")
    # Keep the script running so the viewer stays open
    # Use the is_alive property and render loop as per mujoco-python-viewer examples
    while viewer.is_alive:
        mujoco.mj_step(model, data) # Add simulation step
        viewer.render()

    viewer.close() # Add viewer close

except FileNotFoundError:
    print(f"Error: XML file not found at {xml_filepath}")
    print(f"Current script directory: {script_dir}")
    print("Ensure the path is correct relative to the script location.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the model or launching the viewer: {e}")
    # Print more details if it's a MuJoCo loading error
    # Updated error handling to not rely on model.compiler
    if hasattr(e, 'message') and ("mjASSERT" in e.message or "XML Error" in e.message or "Compile error" in e.message):
        print(f"Check that the assets directory exists in '{xml_dir}' and contains the required STL files.")
    sys.exit(1) 