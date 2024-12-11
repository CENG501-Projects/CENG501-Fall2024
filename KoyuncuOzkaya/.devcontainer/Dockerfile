FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
ENV CUDA_HOME=/usr/local/cuda-11.1/

# Set non-interactive frontend to prevent interactive prompts during package installations
ARG DEBIAN_FRONTEND=noninteractive

# Set SHELL to use bash for the following RUN commands
# Define environment variable for default shell
SHELL ["/bin/bash", "-c"]
ENV SHELL=/bin/bash

# Update system and install necessary packages in a single RUN statement to reduce image layers
# Using "--allow-unauthenticated" since NVIDIA CUDA repository being used by the base image does not have its public GPG key installed
RUN apt update --allow-unauthenticated --allow-insecure-repositories -y && \
    apt install -qq -y --no-install-recommends \
        build-essential \
        sudo \
        git \
        cmake \
        wget \
        unzip \
        python3-opencv \
        # Folloing is for nice-slam
        libopenexr-dev

RUN pip install mmcv-full==1.4.4 mmsegmentation==0.22.1  
RUN pip install timm tqdm thop tensorboard ipdb h5py ipython Pillow==9.5.0 
RUN pip install -U numpy 

# Add user, so the files created in the container are not owned by root
# Following operations needs 'sudo' to be installed

# Add a new user 'ceng' to use instead of 'root' for better security practices
# Configure user with no-password sudo access
ARG USERNAME=ceng
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a group and user with specified UID/GID and add user to sudoers with no password entry
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the default user to the newly created user
USER $USERNAME

# Clean up APT cache to reduce the image size
RUN sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*
