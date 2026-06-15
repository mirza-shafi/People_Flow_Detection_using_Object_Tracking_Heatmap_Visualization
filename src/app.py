import streamlit as st
import os
import cv2
import tempfile
import yaml
import subprocess
from pathlib import Path

# Title and Layout
st.set_page_config(page_title="People Flow Detection", layout="wide")
st.title("🚶‍♂️ People Flow Detection & Heatmap Tracker")

st.markdown("""
Upload a video to analyze the movement of people. The AI will track individuals, 
count how many cross the designated lines, and generate a heatmap of their activity.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# 1. Model Configuration
st.sidebar.subheader("Model Settings")
target_class = st.sidebar.selectbox("Target Class", ["Person (0)"], index=0)

# 2. Line Configuration
st.sidebar.subheader("Line Positions")
upper_line = st.sidebar.slider("Upper Line Position (Ratio)", min_value=0.1, max_value=0.9, value=0.4, step=0.05)
lower_line = st.sidebar.slider("Lower Line Position (Ratio)", min_value=0.1, max_value=0.9, value=0.65, step=0.05)

# 3. Heatmap Configuration
st.sidebar.subheader("Heatmap Settings")
hm_radius = st.sidebar.slider("Blob Radius", min_value=10, max_value=50, value=25, step=5)
hm_opacity = st.sidebar.slider("Opacity", min_value=0.1, max_value=1.0, value=0.6, step=0.1)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a Video file", type=["mp4", "avi", "mov"])

if st.button("Run Processing") and uploaded_file is not None:
    # Set up temp and output paths
    temp_dir = tempfile.mkdtemp()
    source_path = os.path.join(temp_dir, uploaded_file.name)
    target_path = "output/output_people_flow.mp4"
    heatmap_path = "output/final_heatmap.png"
    
    # Write uploaded video to temp file
    with open(source_path, "wb") as f:
        f.write(uploaded_file.read())
        
    # Generate dynamic config.yaml
    config_data = {
        'video': {
            'source_path': source_path,
            'target_path': target_path,
            'heatmap_image_path': heatmap_path
        },
        'model': {
            'weights_path': 'yolov8n.pt',
            'target_class_id': 0
        },
        'tracker': {
            'type': 'bytetrack'
        },
        'lines': {
            'upper_line_y_ratio': upper_line,
            'lower_line_y_ratio': lower_line
        },
        'heatmap': {
            'radius': hm_radius,
            'opacity': hm_opacity
        }
    }
    
    config_path = os.path.join(temp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    st.info("Starting Processing... this may take a few minutes depending on the video length.")
    
    # Run the main.py pipeline as a subprocess
    with st.spinner("Analyzing Video..."):
        try:
            # Note: Assuming src/ is in PYTHONPATH or we run python src/main.py directly
            # using the local Python environment.
            env = os.environ.copy()
            env["PYTHONPATH"] = "src"
            
            import sys
            result = subprocess.run(
                [sys.executable, "src/main.py", "--config", config_path],
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                st.error("An error occurred during processing.")
                st.code(result.stderr)
            else:
                st.success("Processing Complete!")
                
                # We need to convert the mp4 to web-compatible format if it isn't already.
                # OpenCV writes mp4 using standard codecs, but sometimes browsers prefer h264.
                # Assuming standard mp4 works for simplicity.
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Processed Video")
                    if os.path.exists(target_path):
                        # Browsers need h264 to play mp4s properly from OpenCV
                        h264_path = target_path.replace(".mp4", "_h264.mp4")
                        try:
                            from moviepy import VideoFileClip
                            clip = VideoFileClip(target_path)
                            clip.write_videofile(h264_path, codec="libx264", audio=False, logger=None)
                            st.video(h264_path)
                        except Exception as e:
                            st.warning(f"Failed to convert video for web playback. Attempting raw file...")
                            st.video(target_path)
                        
                        with open(target_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Raw Output Video",
                                data=file,
                                file_name="output_people_flow.mp4",
                                mime="video/mp4"
                            )
                    else:
                        st.warning("Processed video not found.")
                        
                with col2:
                    st.subheader("Static Heatmap")
                    if os.path.exists(heatmap_path):
                        st.image(heatmap_path, use_container_width=True)
                    else:
                        st.warning("Heatmap not found.")
                        
        except Exception as e:
            st.error(f"Failed to execute pipeline: {e}")
            
elif uploaded_file is None:
    st.warning("Please upload a video to begin.")
