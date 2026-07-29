#!/bin/bash
export GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

export MODEL_NAME="Manojb/stable-diffusion-2-1-base"
export OUTPUT_DIR="checkpoints/Silica-$(date +%Y-%m-%d_%H-%M-%S)"

# Dataset Base Directories
export TRAIN_DATA_DIR_HYPERSIM="datasets/hypersim"
export TRAIN_DATA_DIR_GSDS="datasets/gsds_dataset"
export TRAIN_DATA_DIR_GDD="datasets/gdd_dataset"
export TRAIN_DATA_DIR_TRANS10K="datasets/trans10k"
export TRAIN_DATA_DIR_REF3D="datasets/3dref"
export TRAIN_DATA_DIR_M18K="datasets/mirage18k"
export TRAIN_DATA_DIR_GWD="datasets/gw_depth"

export RESOLUTION=768
export NORMTYPE="trunc_disparity"
export MIXED_PRECISION="fp16"

export BATCH_SIZE=7
export GAS=2
export LR=3e-5
export STEPS=20000
export TIMESTEP=999 
export TASK_NAME="glass_segmentation+depth"

mkdir -p $OUTPUT_DIR

uv run accelerate launch \
  --num_processes=1 \
  --mixed_precision=$MIXED_PRECISION \
  -m silica_core.train \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --learning_rate=$LR \
  --lr_scheduler="constant" \
  --segmentation_task="predict glass segmentation" \
  --depth_task="predict depth map" \
  --resolution=$RESOLUTION \
  --norm_type=$NORMTYPE \
  --train_data_dir_hypersim=$TRAIN_DATA_DIR_HYPERSIM \
  --train_data_dir_gsds=$TRAIN_DATA_DIR_GSDS \
  --train_data_dir_gdd=$TRAIN_DATA_DIR_GDD \
  --train_data_dir_trans10k=$TRAIN_DATA_DIR_TRANS10K \
  --train_data_dir_ref3d=$TRAIN_DATA_DIR_REF3D \
  --train_data_dir_m18k=$TRAIN_DATA_DIR_M18K \
  --train_data_dir_gwd=$TRAIN_DATA_DIR_GWD \
  --lr_warmup_steps=0 \
  --max_train_steps=$STEPS \
  --timestep=$TIMESTEP \
  --task_name=$TASK_NAME \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=0 \
  --validation_steps=1000 \
  --allow_tf32 \
  --seed=42 \
  --random_flip