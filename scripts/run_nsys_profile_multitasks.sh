#!/bin/bash
export cuda_visible_devices=0,1
project_root="/mnt"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

cd $project_root
nsys profile \
  -o ./profile/profile_result/nsys_profile_multitasks123 \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --python-sampling=true \
  --sample=cpu \
  --force-overwrite=true \
  --stats=true \
  --cuda-memory-usage=true \
  torchrun --nproc_per_node=2 scripts/inference_batch/multi_tasks.py \
    --model_path ./models/BAGEL-7B-MoT \
    --tasks ./data_profile/multi_tasks/tasks_mixed.json \
    --output ./results/multi_tasks \
    --seed 42

echo ""
echo "============================================"
echo "Profiling finished"