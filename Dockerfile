FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /workspace/app
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace/
RUN git clone --depth 1 https://github.com/pytorch/audio
WORKDIR /workspace/audio
RUN sed -i '1i#include "float.h"' src/libtorchaudio/cuctc/src/ctc_prefix_decoder_kernel_v2.cu
RUN python setup.py develop --no-deps

RUN git clone --recursive --depth 1 https://github.com/pytorch/kineto && \
    cd kineto/libkineto && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

WORKDIR /workspace/app