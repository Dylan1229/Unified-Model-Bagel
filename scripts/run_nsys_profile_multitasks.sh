#!/bin/bash
export cuda_visible_devices=1,2,3,4
project_root="/mnt"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

cd $project_root
nsys profile \
  -o ./results/reports/nsys/nsys_profile_multitasks \
  --trace=cuda,nvtx,osrt,cudnn \
  --python-sampling=true \
  --sample=cpu \
  --force-overwrite=true \
  --stats=true \
  --cuda-memory-usage=true \
  torchrun --nproc_per_node=2 scripts/inference_multi/multi_tasks_main.py \
    --model_path ./models/BAGEL-7B-MoT \
    --tasks ./data_profile/multi_tasks/tasks_mixed.json \
    --output ./results/multi_tasks \
    --seed 42 \
    --parallel_mode data_parallel