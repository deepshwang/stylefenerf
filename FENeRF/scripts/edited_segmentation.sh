export CUDA_VISIBLE_DEVICES=${1:-0}
EDIT_TYPE=${2:-surprised}
EXP_DIR=${3:-/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs}
ROOT_DIR=$EXP_DIR/${EDIT_TYPE}/edited
SAVE_DIR=$EXP_DIR/${EDIT_TYPE}/segmaps

python prepare_segmaps.py --root_dir $ROOT_DIR --save_dir $SAVE_DIR