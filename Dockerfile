FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace1

RUN  apt-get update \
  && apt-get install -y python3 python3-pip \
  && apt-get install -y wget git

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch==2.0.0 torchvision cython scipy pycocotools tqdm numpy==1.23 opencv-python torch_geometric