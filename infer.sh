
export CUDA=0

export CHECKPOINT_DIR="silica-mirage/silica-v1-0"
export INPUT_IMAGES="input/image_dir"
export OUTPUT_DIR="output_vis"
export INTRINSICS_TXT="example/intrinsics.txt"

CUDA_VISIBLE_DEVICES=$CUDA uv run python infer.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --seed=42 \
        --input_dir=$INPUT_IMAGES \
        --output_dir=$OUTPUT_DIR \
        --intrinsics_file=$INTRINSICS_TXT

