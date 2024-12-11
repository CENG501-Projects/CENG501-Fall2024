#!/usr/bin/env bash

# This script should be executed with sudo privileges

cd CompletionFormer/src/model/deformconv/
CUDA_HOME=/usr/local/cuda-11.1/ python setup.py build install

export COMMIT=4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
wget https://github.com/NVIDIA/apex/archive/${COMMIT}.zip
unzip ${COMMIT}.zip
rm ${COMMIT}.zip
cd apex-${COMMIT}
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./