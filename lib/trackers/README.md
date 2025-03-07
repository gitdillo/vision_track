#Trackers


To add a new tracker:

- Create a new file in the trackers directory (e.g., new_tracker.py)

- Implement the new tracker class, inheriting from TrackingAlgorithmBase (see example files under directory "trackers" for implementation)

- Override the _create_tracker method (and any other methods that need custom implementation)

- In the config file use the name of the tracker (as it appears in the filename under directory "trackers"), e.g.

{

  "tracking_algorithm": "MedianFlowTracker",

  ...

}
