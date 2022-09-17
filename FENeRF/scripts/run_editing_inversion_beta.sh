export CUDA_VISIBLE_DEVICES=${1:-0}
GENERATOR_PATH=${2:-/home/nas4_user/sungwonhwang/logs/FENeRF/pretrained_models/315000_generator.pth}
IMAGE_PATH=${3:-/home/nas4_user/sungwonhwang/logs/FENeRF/render/e4e_inverted_images/RGB}
SEG_PATH=${4:-/home/nas4_user/sungwonhwang/logs/FENeRF/render/StyleCLIP_edit/surprised/inference_results/segmaps}
SAVE_DIR=${5:-/home/nas4_user/sungwonhwang/logs/FENeRF/render/StyleCLIP_edit/surprised/original_editing_method}
CHECKPOINT_PATH=${6:-/home/nas4_user/sungwonhwang/logs/FENeRF/render/e4e_inverted_images}
FACE_ANGLE_PATH=${7:-/home/nas4_user/sungwonhwang/logs/FENeRF/outputs/face_angles_8.pkl}
EXP_NAME=00001
echo Inverting with...
echo $IMAGE_PATH
echo $SEG_PATH

python inverse_render_double_semantic.py $EXP_NAME \
                                         $GENERATOR_PATH \
                                         --image_path $IMAGE_PATH \
                                         --seg_path $SEG_PATH \
                                         --save_dir $SAVE_DIR \
                                         --background_mask \
                                         --image_size 128 \
                                         --latent_normalize \
                                         --lambda_seg 1. \
                                         --lambda_img 0.2 \
                                         --lambda_percept 1. \
                                         --lock_view_dependence True \
                                         --load_checkpoint True \
                                         --checkpoint_path $CHECKPOINT_PATH \
                                         --face_angle_path $FACE_ANGLE_PATH 