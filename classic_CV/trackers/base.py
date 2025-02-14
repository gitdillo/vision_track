# trackers/base.py

import cv2
import numpy as np


class TrackingAlgorithmBase:
    def __init__(self):
        self.trackers = cv2.legacy.MultiTracker_create()
        self.object_ids = []
        self.next_object_id = 0

        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi = None
        self.last_x = 0
        self.last_y = 0


    def select_ROI(self, frame):
        self.frame = frame
        self.display_frame = frame.copy()
        self.window_name = "Select ROI"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_events)

        print("Controls:")
        print("Use mouse wheel to zoom in/out.")
        print("Click once to start drawing, click again to finish.")
        print("Press 'c' to confirm selection, 'r' to reset, 'q' to quit.")

        while True:
            display = self.get_zoomed_frame()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Quit without selecting ROI
                break
            elif key == ord("c") and self.roi:  # Confirm selection
                cv2.destroyWindow(self.window_name)
                return self.roi
            elif key == ord("r"):  # Reset selection
                self.reset_selection()

        cv2.destroyWindow(self.window_name)
        return None

    def mouse_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = self.zoom_factor
            if flags > 0:  # Scroll up, zoom in
                self.zoom_factor *= 1.1
            else:  # Scroll down, zoom out
                self.zoom_factor = max(1.0, self.zoom_factor / 1.1)

            # Adjust offset to keep the mouse position fixed
            self.offset_x += (x / old_zoom) - (x / self.zoom_factor)
            self.offset_y += (y / old_zoom) - (y / self.zoom_factor)

            self.clamp_offset()

        elif event == cv2.EVENT_LBUTTONUP:
            if not self.drawing:
                self.start_point = self.screen_to_image(x, y)
                self.end_point = self.start_point
                self.drawing = True
            else:
                self.end_point = self.screen_to_image(x, y)
                self.update_roi()
                self.drawing = False
            self.update_display()

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = self.screen_to_image(x, y)
            self.update_display()

    def screen_to_image(self, x, y):
        return (
            int(x / self.zoom_factor + self.offset_x),
            int(y / self.zoom_factor + self.offset_y),
        )

    def clamp_offset(self):
        h, w = self.frame.shape[:2]
        self.offset_x = max(0, min(self.offset_x, w - w / self.zoom_factor))
        self.offset_y = max(0, min(self.offset_y, h - h / self.zoom_factor))

    def get_zoomed_frame(self):
        h, w = self.frame.shape[:2]
        new_w, new_h = int(w / self.zoom_factor), int(h / self.zoom_factor)
        x1, y1 = int(self.offset_x), int(self.offset_y)
        x2, y2 = min(w, x1 + new_w), min(h, y1 + new_h)
        cropped = self.display_frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def update_roi(self):
        if self.start_point and self.end_point:
            x1, y1 = (
                min(self.start_point[0], self.end_point[0]),
                min(self.start_point[1], self.end_point[1]),
            )
            x2, y2 = (
                max(self.start_point[0], self.end_point[0]),
                max(self.start_point[1], self.end_point[1]),
            )
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                self.roi = (x1, y1, w, h)

    def update_display(self):
        self.display_frame = self.frame.copy()
        if self.start_point and self.end_point:
            cv2.rectangle(
                self.display_frame, self.start_point, self.end_point, (0, 255, 0), 2
            )

    def reset_selection(self):
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi = None
        self.display_frame = self.frame.copy()



    def initialize(self, frame, bounding_boxes):
        frame = self._ensure_bgr(frame)
        print(f"Frame type: {frame.dtype}, shape: {frame.shape}")

        initialization_successful = False
        for bbox in bounding_boxes:
            print(f"Bounding box: {bbox}, type: {type(bbox)}")

            x, y, w, h = tuple(map(int, bbox))
            if (
                w <= 0
                or h <= 0
                or x < 0
                or y < 0
                or x + w > frame.shape[1]
                or y + h > frame.shape[0]
            ):
                print("Invalid bounding box: out of bounds or negative dimensions.")
                continue

            tracker = self._create_tracker()
            try:
                success = self.trackers.add(tracker, frame, (x, y, w, h))
                print(f"Tracker add success: {success}")
                if success:
                    self.object_ids.append(self.next_object_id)
                    self.next_object_id += 1
                    initialization_successful = True
            except Exception as e:
                print(f"Error adding tracker: {e}")

        if not initialization_successful:
            print("No valid trackers were initialized.")

        return initialization_successful

    def update(self, frame):
        frame = self._ensure_bgr(frame)
        success, boxes = self.trackers.update(frame)
        tracked_objects = {}

        if success:
            for i, box in enumerate(boxes):
                x, y, w, h = [int(v) for v in box]
                tracked_objects[self.object_ids[i]] = (x, y, w, h)

        return tracked_objects

    def add_object(self, frame, bounding_box):
        tracker = self._create_tracker()
        success = self.trackers.add(tracker, frame, tuple(bounding_box))
        if success:
            self.object_ids.append(self.next_object_id)
            self.next_object_id += 1
        return success

    def remove_object(self, object_id):
        if object_id in self.object_ids:
            index = self.object_ids.index(object_id)
            del self.object_ids[index]
            print(f"Object ID {object_id} removed. Reinitialization required.")

    def handle_disappearance(self):
        # Placeholder implementation
        pass

    def get_tracked_objects(self):
        return {
            obj_id: bbox
            for obj_id, bbox in zip(self.object_ids, self.trackers.getObjects())
        }

    @staticmethod
    def _ensure_bgr(frame):
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def _create_tracker(self):
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement _create_tracker method")



# Test the implementation
if __name__ == "__main__":
    # Create a dummy frame for testing
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    frame = cv2.putText(
        frame,
        "Test Frame",
        (150, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    tracker_base = TrackingAlgorithmBase()
    roi = tracker_base.select_ROI(frame)

    print("Selected ROI:", roi)