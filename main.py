import argparse
import json
import cv2
import logging
import io
from datetime import datetime
from data_io.handlers import InputHandler, OutputHandler
# from data_io.data_format import DataFormat
from trackers import get_tracker


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
        nargs="?",
        const="",
        help="Output file name (use '-' to disable saving, omit for default datetime name)",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def process_live_camera(config, output_handler):
    input_source = config.get("input_source", 0)
    input_handler = InputHandler(input_source)
    input_handler.warm_up(2)

    frame, _ = input_handler.fetch_frame()

    tracker_name = config.get("tracking_algorithm", "CSRTTracker")
    TrackerClass = get_tracker(tracker_name)
    tracker = TrackerClass()

    logging.info(f"Using tracker: {tracker_name}")

    bbox = tracker.select_ROI(frame)

    if output_handler:
        output_handler.set_roi(frame, bbox)

    if tracker.initialize(frame, [bbox]):
        logging.info("Tracking initialized. Starting main loop...")

        try:
            while True:
                frame, ret = input_handler.fetch_frame()
                if not ret:
                    logging.info("End of video feed or error fetching frame.")
                    break

                tracked = tracker.update(frame)

                for obj_id, bbox in tracked.items():
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID: {obj_id}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if output_handler:
                    output_handler.write_frame(
                        frame,
                        [
                            {"bbox": bbox, "confidence": 1.0}
                            for bbox in tracked.values()
                        ],
                    )

        finally:
            input_handler.release()
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
    if args.output != "-":
        if args.output:
            output_file = (
                f"{args.output}.zip"
                if not args.output.endswith(".zip")
                else args.output
            )
        else:
            output_file = f"{datetime.now().strftime('%Y%m%d-%H_%M_%S')}.zip"

        fps = 30  # Default FPS, you might want to get this from the input source
        frame_size = (
            640,
            480,
        )  # Default frame size, you might want to get this from the input source
        output_handler = OutputHandler(output_file, fps, frame_size)

    if isinstance(config.get("input_source", 0), str) and config[
        "input_source"
    ].endswith(".zip"):
        logging.info("Saved data processing not implemented yet")
    else:
        process_live_camera(config, output_handler)

    if output_handler:
        output_handler.add_file("console.log", log_handler.get_contents())
        output_handler.finalize()
        logging.info(f"Output saved to {output_handler.output_path}")


if __name__ == "__main__":
    main()