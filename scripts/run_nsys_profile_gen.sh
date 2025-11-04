#!/bin/bash
# export cuda_visible_devices=0
project_root="/workspace/Bagel"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

cd $project_root
nsys profile \
  -o ./profile/profile_result/nsys_profile_gen_$(date +%Y%m%d_%H%M%S) \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --python-sampling=true \
  --sample=cpu \
  --force-overwrite=true \
  --stats=true \
  --cuda-memory-usage=true \
  python .results/reports/nsys/image_gen.py \
    --model_path ./models/BAGEL-7B-MoT \
    --prompt_pth ./data_profile/prompt/text2image.csv \
    --output ./results/image_gen \
    --seed 42 \
    --think True \
    --do_sample False \
    --image_shapes 768 768


echo ""
echo "============================================"
echo "Profiling finished"

# In the docker /workspace/Bagel, run 'bash profile/run_nsys_profile_gen_batch.sh' 
