#!/bin/bash

# install dev tools
apt install -y libgoogle-glog-dev libboost-all-dev

# unarchive cusparselt
cd cusparselt && {
  echo $PWD && echo "unarchiving cusparselt"
  tar -xf libcusparse_lt-linux-x86_64-0.3.0.3-archive.tar.xz
  tar -xf libcusparse_lt-linux-x86_64-0.4.0.7-archive.tar.xz
  tar -xf libcusparse_lt-linux-x86_64-0.5.0.1-archive.tar.xz
  tar -xf libcusparse_lt-linux-x86_64-0.5.2.1-archive.tar.xz
} && cd ..

# build Samoyeds Kernel
GPU_CC=$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader)

if [ "$GPU_CC" = "8.0" ]; then
    CUDA_COMPUTE_CAPABILITY=80
elif [ "$GPU_CC" = "8.6" ]; then
    CUDA_COMPUTE_CAPABILITY=86
elif [ "$GPU_CC" = "8.9" ]; then
    CUDA_COMPUTE_CAPABILITY=89
elif [ "$GPU_CC" = "9.0" ]; then
    CUDA_COMPUTE_CAPABILITY=90
else
    echo "Unsupported GPU compute capability: $GPU_CC"
    exit 1
fi

mkdir -p build
cd build && {
  echo "Building Samoyeds-Kernel"
  rm -rf *
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="$CUDA_COMPUTE_CAPABILITY"
  make -j
} && cd ..

# build baselines
cd benchmark/third_party/venom && {
  ./scripts/build.sh
} && cd ../../../
