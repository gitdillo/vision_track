# File: vision_track/ML_training/__init__.py

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Now we can import from classic_CV
from classic_CV.data_io.handlers import InputHandler, OutputHandler
from classic_CV.data_io.data_format import DataFormat

__all__ = ["InputHandler", "OutputHandler", "DataFormat"]
