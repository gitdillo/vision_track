# File: vision_track/__init__.py
from . import classic_CV
from . import ML_training

# Optionally, you can re-export some modules for easier access
from .classic_CV.data_io.data_format import DataFormat
from .classic_CV.data_io.handlers import InputHandler, OutputHandler

__all__ = ['classic_CV', 'ML_training', 'DataFormat', 'InputHandler', 'OutputHandler']
