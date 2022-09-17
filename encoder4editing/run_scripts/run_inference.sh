export CUDA_VISIBLE_DEVICES=${1:-0}
IMAGES_DIR=${2:-/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_orig}
SAVE_DIR=${3:-/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs}
PRETRAINED=${4:-/home/nas4_user/sungwonhwang/logs/encoder4editing/pretrained_models/e4e_ffhq_encode.pt}

python scripts/inference.py \
--images_dir=$IMAGES_DIR \
--save_dir=$SAVE_DIR \
$PRETRAINED
