# ML_training

This directory contains resources used for training machine learning algorithms.

It is expected that the training data for these will be provided by the scripts in directory "classic_CV". The output format for this training data will be described there.

This training directory has had the following setup for Linux 24.04:
1. Installed python3.11 (python 3.12 had version problems between pandas and MLflow) using:

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.11 python3.11-venv

2. Created and activated a virtual environment for python 3.11 with (these commands were run inside the ML_training directory):

    python3.11 -m venv training_env
    source training_env/bin/activate


3. Installed the following. Use the path to the python inside the virtual environment, rather than pip directly, to ensure things are correctly installed:

    /path/to/vision_track/ML_training/training_env/bin/python -m pip install tensorflow torch torchvision opencv-python-headless matplotlib numpy scikit-learn pandas mlflow