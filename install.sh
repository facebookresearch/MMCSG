# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

conda create --name chime_recipe python=3.11
conda activate chime_recipe
conda config --add channels nvidia
conda config --add channels pytorch
# requirements.txt contains also install of pytorch, 
# check that it corresponds to your system and version of cuda
conda install --file requirements.txt
pip install Cython
pip install nemo_toolkit[all]==1.21

pushd .
cd tools/multitalker_wer
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    -I $CONDA_PREFIX/include \
    -I $CONDA_PREFIX/include/eigen3 \
    -L $CONDA_PREFIX/lib \
    -Wl,-rpath=$CONDA_PREFIX/lib \
    MultiTalkerEditDistance.cpp MultiTalkerWordErrorRate.cpp UnicodeNorm.cpp multitalker_wer_pybind.cpp \
    -lfolly -lglog -lfmt -licudata -licuio -licui18n -licuuc \
    -o multitalker_wer_pybind$(python3-config --extension-suffix) 
popd

# you might want to add these to your ~/.bashrc or set them before running the system
export PYTHONPATH=$PWD/tools/multitalker_wer:$PYTHONPATH # necessary for evaluation step
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python # necessary for finetuning step
