import importlib
import os

# Dynamically import all tracker modules
tracker_modules = {}
for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # Remove .py extension
        tracker_modules[module_name] = importlib.import_module(
            f"classic_CV.trackers.{module_name}", package="vision_track"
        )


def get_tracker(tracker_name):
    for module in tracker_modules.values():
        if hasattr(module, tracker_name):
            return getattr(module, tracker_name)
    raise ValueError(f"Tracker '{tracker_name}' not found")


# Explicitly export the get_tracker function
__all__ = ["get_tracker"]