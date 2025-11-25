#!/bin/bash

# Usage: 
#   ./run_docker.sh              # Use all GPU
#   ./run_docker.sh 0            # use GPU 0
#   ./run_docker.sh 2,3,4        # use GPU 2, 3, 4

# Default to use all GPU, if provided parameters, use the specified GPU
if [ -z "$1" ]; then
  GPU_ARG="all"
else
  GPU_ARG="\"device=$1\""
fi

echo "Using GPUs: $GPU_ARG"

docker run --gpus $GPU_ARG -it --rm \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --cap-add=PERFMON \
  --security-opt seccomp=unconfined \
  --shm-size=64g \
  -v /scr/dataset/yuke/fanjiang/repo/unified-model/Bagel:/workspace/Bagel \
  -w /workspace/Bagel \
  bagel-profile:latest \
  /bin/bash
