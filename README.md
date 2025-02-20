# vision_track

Directory "classic_CV" contains resources for tracking a feature using classic computer vision algorithms.

Directory "ML_training" contains resources for training machine learning algorithms, using the output of the scripts in directory "classic_CV" as training data.


A docker container handles the requirements for the environment (mostly the somewhat annoying training environment demended by "ML_training").

Before proceeding with Docker, install the NVidia Container Toolkit as described here:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

TL:DR, run the following:

`curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list`

`sudo apt-get update`

`sudo apt-get install -y nvidia-container-toolkit`


Install docker:

`sudo apt-get install docker.io docker-buildx`

Add user to the docker group:

`sudo usermod -aG docker $USER`

`newgrp docker`


In the the root directory of the project there should be a file named `Dockerfile`. From this directory, build the container with:

`docker build -t vision_track_container .`

NOTE: It might be necessary to configure Docker to use the NVidia runtime, in which case issue the following:
`sudo nvidia-ctk runtime configure --runtime=docker`

`sudo systemctl restart docker`


From the project root directory (vision_track/), run the container with:

`docker run --gpus all -it -v $(pwd):/vision_track -w /vision_track --device-cgroup-rule='c 81:* rmw'  -v /dev:/dev vision_track_container`