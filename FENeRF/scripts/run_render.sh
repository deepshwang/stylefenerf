export CUDA_VISIBLE_DEVICES=${1:-0}
OUTPUT_DIR=${2:-/home/nas4_user/sungwonhwang/logs/FENeRF/outputs}
NUM_SAMPLES=${3:-300}
GENERATOR_PATH=${4:-/home/nas4_user/sungwonhwang/logs/FENeRF/pretrained_models/315000_generator.pth}

python render_multiview_images_double_semantic.py $GENERATOR_PATH \
                                                  --curriculum CelebA_double_semantic_texture_embedding_256_dim_96 \
                                                  --seeds $NUM_SAMPLES \
                                                  --output_dir $OUTPUT_DIR