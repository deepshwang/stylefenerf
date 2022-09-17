export CUDA_VISIBLE_DEVICES=${1:-0}
ROOT_DIR='/home/nas4_user/sungwonhwang/logs/FENeRF/render/e4e_inverted_images/RGB'
SAVE_DIR='/home/nas4_user/sungwonhwang/logs/FENeRF/render/e4e_inverted_images/SEG'

python prepare_segmaps.py --root_dir $ROOT_DIR --save_dir $SAVE_DIR