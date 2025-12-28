#!/bin/bash
export cuda_visible_devices=4,5,6,7

project_root="/scr/dataset/yuke/fanjiang/repo/unified-model/Bagel"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

cd $project_root

python inference/mixed_tasks.py \
  --model_path ./models/BAGEL-7B-MoT \
  --tasks ./inference/tasks/mixed_tasks/und-gen.json \
  --output ./results/mixed-tasks \
  --num_gpus 1  \
  --seed 42 \
