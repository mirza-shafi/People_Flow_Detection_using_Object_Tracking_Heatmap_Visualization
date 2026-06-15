FROM python:3.12-slim

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \\
    libgl1 \\
    libglib2.0-0 \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY output/ ./output/

# Download YOLO weights to the root directory
RUN wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt -O yolov8n.pt

# Entry point
CMD ["python", "src/main.py"]
