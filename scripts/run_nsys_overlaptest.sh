#!/bin/bash

# CHANGE THIS: Use 0 (the first visible GPU in your container)
export CUDA_VISIBLE_DEVICES=0

project_root="/mnt"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

# MPS Setup
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-user-$USER
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-user-$USER
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY

echo "Starting NVIDIA MPS on GPU $CUDA_VISIBLE_DEVICES..."
nvidia-cuda-mps-control -d

# Execution
cd $project_root

# NOTE: Using 'python' instead of torchrun.
# NOTE: Keeping 38GiB limit to fit 2 models on one 80GB GPU.
nsys profile \
  -o ./results/reports/nsys/overlap_512 \
  --trace=cuda,nvtx,osrt,cudnn \
  --python-sampling=true \
  --sample=cpu \
  --force-overwrite=true \
  --stats=true \
  --cuda-memory-usage=true \
  python inference/inference_multi/overlap.py \
    --model_path ./models/BAGEL-7B-MoT \
    --tasks ./data_profile/multi_tasks/overlap_512.json \
    --output ./results/multi_tasks \
    --seed 42 \
    --max_mem_per_gpu "80GiB" \
    --device 0

# MPS Teardown
echo "Stopping NVIDIA MPS..."
echo quit | nvidia-cuda-mps-control

# Cleanup
pkill -f "overlap.py"