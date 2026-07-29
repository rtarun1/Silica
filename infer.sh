export CUDA=0

export CHECKPOINT_DIR="rtarun1/silica-v1-0"
export INPUT_IMAGES="example/images"
export INPUT_DEPTH="example/depth" # raw sensor depth
export OUTPUT_DIR="example/output"
export INTRINSICS_TXT="example/intrinsics.txt"

# reduce resolution if cuda memory is insufficient or need faster inference, but this may reduce the quality of the output
export RESOLUTION=768

CUDA_VISIBLE_DEVICES=$CUDA uv run python -m silica_core.infer \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --seed=42 \
        --input_dir=$INPUT_IMAGES \
        --processing_res=$RESOLUTION \
        --input_sensor_depth_dir=$INPUT_DEPTH \
        --output_dir=$OUTPUT_DIR \
        --intrinsics_file=$INTRINSICS_TXT

