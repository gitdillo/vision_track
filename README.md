# vision_track

Directory "classic_CV" contains resources for tracking a feature using classic computer vision algorithms.

Directory "ML_training" contains resources for training machine learning algorithms, using the output of the scripts in directory "classic_CV" as training data.


A docker container handles the requirements for the environment (mostly the somewhat annoying training environment demended by "ML_training").

Before proceeding with Docker, install the NVidia Container Toolkit as described here:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html


Install docker:

sudo apt-get install docker.io docker-buildx-plugin

In the the root directory of the project there should be a file named `Dockerfile`. From this directory, build the container with:

`docker build -t vision_track_container .`

<<<<<<< HEAD
NOTE: It might be necessary to configure Docker to use the NVidia runtime, in which case issue the following:
=======
It might be necessary to configure Docker to use the NVidia runtime, in which case issue the following:

>>>>>>> 6439a2b96d2bbdcc334427790309a4da0bec244a
`sudo nvidia-ctk runtime configure --runtime=docker`

`sudo systemctl restart docker`


Run the container with:
<<<<<<< HEAD
`docker run --gpus all -it -v $(pwd):/vision_track vision_track_container`
=======

`docker run --gpus all -it -v $(pwd):/workspace vision_track_container`
>>>>>>> 6439a2b96d2bbdcc334427790309a4da0bec244a
