# People Flow Detection

This project uses YOLOv8 and ByteTrack to detect, track, and count people crossing designated zones in a video. It also generates a heatmap visualization of the flow of people throughout the entire video.

## Features
- **Object Detection**: Uses state-of-the-art YOLOv8 to accurately detect people.
- **Object Tracking**: Uses ByteTrack for robust multi-object tracking.
- **Line Crossing Counting**: Tracks how many people cross two defined lines (e.g., an "IN" line and an "OUT" line) with a real-time dashboard overlay.
- **Heatmap Generation**: Generates an aggregated static heatmap showing the areas with the most foot traffic over the duration of the video.

## Project Structure
```text
.
├── config/
│   └── config.yaml     # Application configuration (paths, model settings, thresholds)
├── data/               # Directory for input video files
├── output/             # Directory for output videos and generated heatmaps
├── src/
│   ├── config.py       # Configuration loader
│   ├── main.py         # Entry point for the application
│   ├── tracker.py      # Core detection and tracking logic using YOLO and ByteTrack
│   └── visualizer.py   # Handles all supervision annotations (boxes, lines, heatmaps)
├── Dockerfile          # Docker configuration for containerized deployment
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Quick Start

### 1. Local Development (Python Virtual Environment)

**Prerequisites:** Python 3.10+

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your input video in the `data/` folder (or use the configured default).
5. Run the application:
   ```bash
   PYTHONPATH=src python src/main.py
   ```
   *Optional*: Override the input video via CLI:
### 2. Streamlit Web Dashboard (UI)

If you prefer to use a visual interface to upload videos and tweak parameters interactively:

1. Ensure your virtual environment is active and dependencies are installed (including `streamlit`).
2. Run the Streamlit app:
   ```bash
   PYTHONPATH=src streamlit run src/app.py
   ```
3. A web browser will automatically open at `http://localhost:8501`.

### 3. Docker Deployment

**Prerequisites:** Docker installed and running.

1. Build the Docker image:
   ```bash
   docker build -t people-flow-detection .
   ```
2. Run the Docker container:
   ```bash
   # Mount the output directory to get the files back on your host machine
   docker run -v $(pwd)/output:/app/output people-flow-detection
   ```

## Configuration
All major parameters are configurable via `config/config.yaml`.
- **`video`**: Set paths for the input video, output video, and the final heatmap image.
- **`model`**: Change the YOLO weights path or the target detection class.
- **`lines`**: Adjust the vertical position (ratio) of the counting lines on the screen.
- **`heatmap`**: Adjust the radius and opacity of the heatmap blobs.

## Outputs
When the pipeline completes, check the `output/` folder for:
1. `output_people_flow.mp4`: The annotated video with tracking boxes and live counting.
2. `final_heatmap.png`: The static heatmap generated from all tracks in the video.
