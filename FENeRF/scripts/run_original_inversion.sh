export CUDA_VISIBLE_DEVICES=${1:-0}
EXP_NAME=${2:-00001}
GENERATOR_PATH=/home/nas4_user/sungwonhwang/logs/FENeRF/pretrained_models/315000_generator.pth
IMAGE_PATH=/home/nas4_user/sungwonhwang/logs/FENeRF/render/e4e_inverted_images/RGB
SEG_PATH=/home/nas4_user/sungwonhwang/logs/FENeRF/render/e4e_inverted_images/SEG
SAVE_DIR=/home/nas4_user/sungwonhwang/logs/FENeRF/render/e4e_inverted_images
echo Inverting with...
echo $IMAGE_PATH
echo $SEG_PATH

python inverse_render_double_semantic.py $EXP_NAME \
                                         $GENERATOR_PATH \
                                         --image_path $IMAGE_PATH \
                                         --seg_path $SEG_PATH \
                                         --save_dir $SAVE_DIR \
                                         --image_size 128 \
                                         --latent_normalize \
                                         --lambda_seg 1. \
                                         --lambda_img 0.2 \
                                         --lambda_percept 1. \
                                         --lock_view_dependence True \
                                         --recon