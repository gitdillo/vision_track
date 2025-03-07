# /lib

This directory contains mostly class files used in the two parts of the project *classic_CV* and *ML_training*.

Directory `data_io` contains:
- `data_format.py`: the description of the data format used as the output of the *classic_CV* and as the input of `ML_training`.
- `handlers.py`: the classes for the `InputHandler` and `OutputHandler`, used for providing input and output pipelines of frames and metadata to the rest of the code.

Directory `trackers` contains various trackers that can be used for feature detection by the scripts in the *classic_CV* part of the project. Any new trackers must be placed there, see *README* inside the directory.