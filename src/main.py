import argparse
import logging
import cv2
import supervision as sv
from pathlib import Path

from config import load_config
from tracker import PeopleTracker
from visualizer import PeopleFlowVisualizer

def setup_logging():
    \"\"\"Configure logging to console and file.\"\"\"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('people_flow.log')
        ]
    )

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="People Flow Detection and Tracking")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--source", type=str, help="Override source video path")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return

    # Override source if provided via CLI
    source_video_path = args.source if args.source else config['video']['source_path']
    target_video_path = config['video']['target_path']
    heatmap_image_path = config['video']['heatmap_image_path']

    if not Path(source_video_path).exists():
        logging.error(f"Source video not found: {source_video_path}")
        return

    logging.info("Initializing Tracker and Visualizer...")
    
    try:
        video_info = sv.VideoInfo.from_video_path(source_video_path)
    except Exception as e:
        logging.error(f"Failed to read video info from {source_video_path}: {e}")
        return

    tracker = PeopleTracker(config)
    visualizer = PeopleFlowVisualizer(config, video_info)

    cumulative_detections = []
    
    logging.info(f"Processing video: {source_video_path}")
    logging.info(f"Target video will be saved to: {target_video_path}")

    # Process Video
    try:
        with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
            for frame in sv.get_video_frames_generator(source_path=source_video_path):
                # Detect and track
                detections = tracker.process_frame(frame)
                
                if len(detections) > 0:
                    cumulative_detections.append(detections)
                    
                # Update visualizer state (lines)
                visualizer.trigger_lines(detections)
                
                # Annotate current frame
                annotated_frame = visualizer.annotate_frame(frame, detections)
                
                # Write to output
                sink.write_frame(frame=annotated_frame)
                
    except Exception as e:
        logging.error(f"An error occurred during video processing: {e}")
        return

    logging.info("Video processing completed. Generating final heatmap...")

    # Generate Heatmap
    try:
        frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
        background_frame = next(frame_generator)
        final_heatmap_img = visualizer.generate_final_heatmap(background_frame, cumulative_detections)
        
        cv2.imwrite(heatmap_image_path, final_heatmap_img)
        logging.info(f"Final heatmap saved to: {heatmap_image_path}")
    except Exception as e:
        logging.error(f"An error occurred while generating the heatmap: {e}")
        return

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
