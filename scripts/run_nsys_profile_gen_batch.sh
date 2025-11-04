#!/bin/bash
export cuda_visible_devices=0
project_root="/mnt"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

cd $project_root
nsys profile \
  -o ./profile/profile_result/nsys_profile_gen_batch \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --python-sampling=true \
  --sample=cpu \
  --force-overwrite=true \
  --stats=true \
  --cuda-memory-usage=true \
  python ./scripts/inference_batch/image_gen_batch.py \
    --model_path ./models/BAGEL-7B-MoT \
    --prompt_pth ./data_profile/prompt/text2image.csv \
    --output ./results/image_gen \
    --do_sample \
    --think True \
    --seed 42





echo ""
echo "============================================"
echo "Profiling finished"