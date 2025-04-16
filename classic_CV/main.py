import argparse
import json
import cv2
import logging
import io
import sys
from datetime import datetime
from vision_track.lib.data_io.handlers import InputHandler, OutputHandler
from vision_track.lib.trackers import get_tracker

class StringIOHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.stream = io.StringIO()

    def emit(self, record):
        msg = self.format(record)
        self.stream.write(f"{msg}\n")

    def get_contents(self):
        return self.stream.getvalue()


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    string_handler = StringIOHandler()
    string_handler.setLevel(logging.INFO)
    logger.addHandler(string_handler)

    return string_handler


def parse_arguments():
    parser = argparse.ArgumentParser(description="Video tracking system")
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path (use '-' for stdout, .zip extension will be added if missing)"
    )
    return parser.parse_args()



def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def process_live_camera(config, output_handler):
    if output_handler is None:
        raise ValueError("Output handler required for camera processing")
    input_source = config.get("input_source", 0)
    input_handler = InputHandler(input_source)
    input_handler.warm_up(2)

    # Get initial frame for ROI selection
    frame, _ = input_handler.fetch_frame()
    if frame is None:
        logging.error("Failed to capture initial frame")
        return

    # Initialize video writers with actual frame parameters
    frame_size = (frame.shape[1], frame.shape[0])
    fps = input_handler.cap.get(cv2.CAP_PROP_FPS)
    output_handler.initialize_video_writers(frame_size, fps)

    # Tracker initialization
    tracker_name = config.get("tracking_algorithm", "CSRTTracker")
    TrackerClass = get_tracker(tracker_name)
    tracker = TrackerClass()
    logging.info(f"Using tracker: {tracker_name}")

    # ROI selection and initialization
    bbox = tracker.select_ROI(frame)
    output_handler.set_roi(frame, bbox)  # Save ROI frame

    if tracker.initialize(frame, [bbox]):
        logging.info("Tracking initialized. Starting main loop...")
        try:
            frame_count = 0
            while True:
                # Capture raw frame
                raw_frame, ret = input_handler.fetch_frame()
                if not ret:
                    logging.info("End of video feed or error fetching frame.")
                    break

                # Save raw frame BEFORE processing
                output_handler.write_raw_frame(raw_frame.copy())

                # Process frame
                annotated_frame = raw_frame.copy()
                tracked = tracker.update(annotated_frame)
                annotations = []

                # Draw annotations on the annotated frame
                for obj_id, bbox in tracked.items():
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"ID: {obj_id}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    annotations.append({"bbox": bbox, "confidence": 1.0})

                # Save annotated frame and annotations
                output_handler.write_annotated_frame(annotated_frame)
                output_handler.add_annotation(frame_count, annotations)

                # Display results
                cv2.imshow("Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_count += 1

        finally:
            # Save metadata and clean up
            output_handler.metadata = {
                "frame_count": frame_count,
                "fps": fps,
                "frame_size": frame_size,
                "tracking_algorithm": tracker_name
            }
            input_handler.release()
            output_handler.release()
            cv2.destroyAllWindows()
    else:
        logging.error("Failed to initialize tracker. Exiting.")


def main():
    log_handler = setup_logging()
    args = parse_arguments()
    config = load_config(args.config)

    logging.info("Loaded configuration:")
    logging.info(json.dumps(config, indent=2))

    output_handler = None
    input_source = config.get("input_source", 0)
    
    try:
        # Initialize input handler to get actual video parameters
        input_handler = InputHandler(input_source)
        input_handler.warm_up(2)
        frame, _ = input_handler.fetch_frame()
        if frame is None:
            raise ValueError("Failed to capture initial frame from input source")
        
        # Get dynamic parameters from actual input
        fps = input_handler.cap.get(cv2.CAP_PROP_FPS)
        frame_size = (frame.shape[1], frame.shape[0])
        input_handler.release()

        if args.output != "-":
            # Initialize output handler with dynamic parameters
            output_file = (f"{args.output}.zip" if not args.output.endswith(".zip") 
                          else args.output) if args.output else \
                         f"{datetime.now().strftime('%Y%m%d-%H_%M_%S')}.zip"
            
            output_handler = OutputHandler(output_file)
            output_handler.initialize_video_writers(frame_size, fps)

        if isinstance(input_source, str) and input_source.endswith(".zip"):
            logging.info("Saved data processing not implemented yet")
        else:
            process_live_camera(config, output_handler)

    except Exception as e:
        logging.error(f"Critical error: {str(e)}", exc_info=True)
        raise

    finally:
        # Cleanup resources
        if output_handler:
            output_handler.add_file("console.log", log_handler.get_contents())
            output_handler.finalize()
            logging.info(f"Output saved to {output_handler.output_path}")



if __name__ == "__main__":
    main()