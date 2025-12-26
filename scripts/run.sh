#!/bin/bash
export cuda_visible_devices=0

project_root="/scratch/fy27/repo/sosp26-unified-model/Bagel"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

cd $project_root

python inference/multi_tasks_main.py \
  --model_path ./models/BAGEL-7B-MoT \
  --tasks ./inference/tasks/mixed_tasks/und-gen.json \
  --output ./results/mixed-tasks \
  --num_gpus 1  \
  --seed 42 \
