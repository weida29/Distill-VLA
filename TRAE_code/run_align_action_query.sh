#!/bin/bash
# Run from TRAE_code/ directory: bash run_align_action_query.sh
# New training script for action query alignment

export TOKENIZERS_PARALLELISM=false

data_name="libero_10_no_noops"
current_time=$(date +%Y%m%d_%H%M%S)

torchrun --standalone --nnodes=1 --nproc_per_node=1 finetune_align_action_query.py \
    --vlm_path="../pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" \
    --config_file_path="../pretrained_models/configs" \
    --data_root_dir="../data/libero" \
    --dataset_name="$data_name" \
    --run_root_dir="../runs" \
    --use_minivlm=true \
    --use_film=false \
    --num_images_in_input=2 \
    --use_proprio=true \
    --use_lora=true \
    --use_fz=false \
    --image_aug=true \
    --use_l1_regression=true \
    --num_steps_before_decay=400000 \
    --max_steps=50000 \
    --save_freq=5000 \
    --save_latest_checkpoint_only=false \
    --merge_lora_during_training=true \
    --batch_size=1 \
    --grad_accumulation_steps=8 \
    --learning_rate=2e-4 \
    --lora_rank=64 \
    --use_pro_version=true \
    --use_visual_teacher=true \
    --use_action_query_alignment=true \
    --action_query_alignment_dropout=0.1 \
    --visual_teacher_config="/tmp/Distill-VLA/visual_teacher/Open-GroundingDino/config/cfg_odvg.py" \
    --visual_teacher_checkpoint="/tmp/Distill-VLA/checkpoints/open_gdino_finetuned/checkpoint_best_regular.pth" \
    --action_loss_weight=1.0 \
    --alignment_loss_weight=0.5 \
    --hs_weight=1.0 \
    --ref_weight=1.0 \
    --wandb_project="vla-alignment" \
    --run_id_note="VLA-ActionQueryAlign--${data_name}--${current_time}"
