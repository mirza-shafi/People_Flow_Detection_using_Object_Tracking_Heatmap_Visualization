import logging
from ultralytics import YOLO
import supervision as sv
from typing import Tuple, List
import numpy as np

class PeopleTracker:
    \"\"\"
    Handles the initialization and execution of YOLO object detection and ByteTrack tracking.
    \"\"\"
    def __init__(self, config: dict):
        self.config = config
        weights_path = config['model']['weights_path']
        logging.info(f"Loading YOLO model with weights: {weights_path}")
        self.model = YOLO(weights_path)
        self.tracker = sv.ByteTrack()
        self.target_class_id = config['model']['target_class_id']
        
    def process_frame(self, frame: np.ndarray) -> sv.Detections:
        \"\"\"
        Runs inference on a frame and updates tracking IDs.
        \"\"\"
        # Inference
        results = self.model(frame, verbose=False)[0]
        
        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter for the target class (e.g., person)
        detections = detections[detections.class_id == self.target_class_id]
        
        # Update tracking IDs
        detections = self.tracker.update_with_detections(detections)
        
        return detections
