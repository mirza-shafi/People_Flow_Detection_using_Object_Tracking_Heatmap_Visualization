import cv2
import supervision as sv
from typing import List, Tuple
import numpy as np

class PeopleFlowVisualizer:
    \"\"\"
    Handles all visual annotations for the people flow detection including
    bounding boxes, labels, line crossings, and heatmaps.
    \"\"\"
    def __init__(self, config: dict, video_info: sv.VideoInfo):
        self.config = config
        self.video_info = video_info
        
        # Determine line coordinates based on proportions
        height = video_info.height
        width = video_info.width
        
        upper_y = int(height * config['lines']['upper_line_y_ratio'])
        lower_y = int(height * config['lines']['lower_line_y_ratio'])
        
        upper_start = sv.Point(0, upper_y)
        upper_end = sv.Point(width, upper_y)
        lower_start = sv.Point(0, lower_y)
        lower_end = sv.Point(width, lower_y)
        
        # Initialize Line Zones
        self.line_zone_upper = sv.LineZone(start=upper_start, end=upper_end)
        self.line_zone_lower = sv.LineZone(start=lower_start, end=lower_end)
        
        # Initialize Annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_padding=4, text_scale=0.5)
        
        self.line_annotator_upper = sv.LineZoneAnnotator(
            thickness=3, text_thickness=2, text_scale=0.7, color=sv.Color.GREEN
        )
        self.line_annotator_lower = sv.LineZoneAnnotator(
            thickness=3, text_thickness=2, text_scale=0.7, color=sv.Color.ROBOFLOW
        )
        
        self.heatmap_annotator = sv.HeatMapAnnotator(
            position=sv.Position.CENTER,
            radius=config['heatmap']['radius'],
            opacity=config['heatmap']['opacity']
        )
        
    def trigger_lines(self, detections: sv.Detections):
        \"\"\"Update line crossing counts based on current detections.\"\"\"
        self.line_zone_upper.trigger(detections)
        self.line_zone_lower.trigger(detections)
        
    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        \"\"\"Applies all annotations to a single frame.\"\"\"
        annotated_frame = frame.copy()
        
        # 1. Heatmap
        annotated_frame = self.heatmap_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        
        # 2. Bounding Boxes and Labels
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        
        labels = [f"ID: {track_id}" for track_id in detections.tracker_id] if detections.tracker_id is not None else []
        if labels:
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            
        # 3. Line Zones
        self.line_annotator_upper.annotate(frame=annotated_frame, line_counter=self.line_zone_upper)
        self.line_annotator_lower.annotate(frame=annotated_frame, line_counter=self.line_zone_lower)
        
        # 4. Custom Dashboard
        cv2.rectangle(annotated_frame, (10, 10), (240, 95), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"TOTAL IN: {self.line_zone_upper.in_count}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"TOTAL OUT: {self.line_zone_lower.out_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                    
        return annotated_frame

    def generate_final_heatmap(self, background_frame: np.ndarray, cumulative_detections: List[sv.Detections]) -> np.ndarray:
        \"\"\"Generates a static heatmap image from accumulated detections over the whole video.\"\"\"
        final_heatmap_img = background_frame.copy()
        for tracking_batch in cumulative_detections:
            final_heatmap_img = self.heatmap_annotator.annotate(
                scene=final_heatmap_img, detections=tracking_batch
            )
        return final_heatmap_img
