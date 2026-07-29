#!/bin/bash
export GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

export CHECKPOINT_PATH="rtarun1/silica-v1-0"
export DATASET_BASE_DIR="datasets/mirage18k"
export OUTPUT_DIR="eval_results/silica-v1-0"

export ALIGNMENT="least_square_disparity"

mkdir -p $OUTPUT_DIR

uv run python -m silica_core.eval \
  --pretrained_model_name_or_path=$CHECKPOINT_PATH \
  --dataset_base_dir=$DATASET_BASE_DIR \
  --output_dir=$OUTPUT_DIR \
  --alignment=$ALIGNMENT \
  --generate \
  --seed=42