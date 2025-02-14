# ML_training

This directory contains resources used for training machine learning algorithms.

It is expected that the training data for these will be provided by the scripts in directory "classic_CV". The output format for this training data will be described there.

This training directory has had the following setup for Ubuntu 24.04 using an NVidia GPU.

Before proceeding with setting up the python virtual environment, ensure that the system has the required NVidia packages installed:
```    
sudo apt install nvidia-driver-XXX  (XXX being a modern version in this case 550)
sudo apt install nvidia-cuda-toolkit
```

If the above commands result in packages being installed, make sure to reboot the machine before proceeding.

The installations inside the virtual environment that are described below depend upon these system packages being present.

NOTE: the instructions below are for creating and then performing actions **inside a virtual environment**! This is to avoid problems with the system wide python installation.

1. Installed python3.11 (python 3.12 had version problems between pandas and MLflow) using:

    ```
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.11 python3.11-venv
    ```
    NOTE: installing python3.11 might not be necessary in the future, if version conflicts between pandas and MLflow are resolved.
    
    NOTE2: Ubuntu 24.04 does not offer a package for python3.11, which is why adding the `deadsnakes/ppa` is needed.

2. Created and activated a virtual environment for python 3.11 with (these commands were run inside the ML_training directory):
    ```
    python3.11 -m venv training_env
    source training_env/bin/activate
    ```

3. Installed the following. In some cases, it might be better to use the path to the python inside the virtual environment, rather than pip directly (i.e. `/path/to/vision_track/ML_training/training_env/bin/python -m pip`), to ensure things are correctly installed:
    ```
    pip install tf-nightly

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

    pip install opencv-python-headless matplotlib numpy scikit-learn pandas mlflow
    ```

4. The environment has been captured in the `requirements.txt` file using command `pip freeze > requirements.txt`. To recreate it use `pip install -r requirements.txt`