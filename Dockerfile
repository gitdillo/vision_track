# Use NVIDIA's PyTorch Docker image with CUDA support
FROM pytorch/pytorch:latest
ARG DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies for OpenCV and other tools
RUN apt-get update && apt-get install -y \
    python3-opencv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    matplotlib \
    numpy \
    scikit-learn \
    pandas \
    mlflow

# Set up working directory inside the container
WORKDIR /vision_track

# Set environment variables to properly handle imports
ENV PYTHONPATH="/vision_track:${PYTHONPATH}"
ENV PYTHONPATH="/:${PYTHONPATH}"
ENV VISION_TRACK_ROOT="/vision_track"

# Copy the entire project into the container (optional)
# COPY . /workspace
# Alternatively, mount your project directory as a volume:
# docker run --gpus all -it -v $(pwd):/workspace vision_track_container

# Default command (can be overridden)
CMD ["bash"]

# Install X11 and VNC server
RUN apt-get update && apt-get install -y x11vnc xvfb

# Set up a virtual display
ENV DISPLAY :0

# Start VNC server
CMD ["x11vnc", "-create", "-forever"]
