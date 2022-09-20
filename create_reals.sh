#!/bin/bash
CUDA_DEVICE=${1:-0}
EDIT_TEXT=${2:-"curly_hair"}
GEN_ORIG=${3:-false}
USE_ORIGINAL_METHOD=${4:-false}

#Configs
N_SAMPLES=200
OUTPUT_DIR=/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs

#Pretrained models directory
FENERF_GENERATOR_PATH=/home/nas4_user/sungwonhwang/logs/FENeRF/pretrained_models/315000_generator.pth

mkdir $OUTPUT_DIR

# [1] Create FENeRF samples
cd FENeRF
bash scripts/run_render.sh $CUDA_DEVICE $OUTPUT_DIR $N_SAMPLES $FENERF_GENERATOR_PATH
cd ..