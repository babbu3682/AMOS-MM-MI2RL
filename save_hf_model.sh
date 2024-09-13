#!/bin/bash

# run "accelerate config" first!

CUDA_VISIBLE_DEVICES="3" python /workspace/0.Challenge/MICCAI2024_AMOSMM/M3D/merge_lora_weights_and_save_hf_model.py \
    --model_name_or_path /workspace/0.Challenge/MICCAI2024_AMOSMM/M3D/pretrained_weight/Meta-Llama-3.1-8B-Instruct \
    --model_type llama3 \
    --model_with_lora /workspace/0.Challenge/MICCAI2024_AMOSMM/M3D/LaMed/output/LaMed-finetune-0003/model_with_lora.bin \
    --output_dir /workspace/0.Challenge/MICCAI2024_AMOSMM/M3D/LaMed/output/LaMed-finetune-0003/hf
