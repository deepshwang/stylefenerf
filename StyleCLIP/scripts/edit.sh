export CUDA_VISIBLE_DEVICES=${1:-0}
EDIT_TYPE=${2:-surprised}
EXP_DIR=${3:-/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs}

python edit.py --edit_type $EDIT_TYPE \
               --latent_path ${EXP_DIR}/e4e_latents.pt \
               --exp_dir $EXP_DIR 