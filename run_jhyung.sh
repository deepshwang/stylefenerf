#!/bin/bash
CUDA_DEVICE=${1:-0}
EDIT_TEXT=${2:-"curly_hair"}
GEN_ORIG=${3:-false}
USE_ORIGINAL_METHOD=${4:-false}

#Configs
N_SAMPLES=1000
OUTPUT_DIR=/home/nas2_userG/junhahyung/fenerf/outputs
mkdir $OUTPUT_DIR

#Pretrained models directory
FENERF_GENERATOR_PATH=/home/nas4_user/sungwonhwang/logs/FENeRF/pretrained_models/315000_generator.pth
EE_PATH=/home/nas4_user/sungwonhwang/logs/encoder4editing/pretrained_models/e4e_ffhq_encode.pt


if [ "$GEN_ORIG" == true ]; then
    # [1] Create FENeRF samples
    cd FENeRF
    bash scripts/run_render.sh $CUDA_DEVICE $OUTPUT_DIR $N_SAMPLES $FENERF_GENERATOR_PATH
    cd ..

    # [2] Conduct StyleGAN inversion using e4e
    cd encoder4editing
    bash run_scripts/run_inference.sh $CUDA_DEVICE ${OUTPUT_DIR}/RGB_orig $OUTPUT_DIR $EE_PATH 
    cd ..
fi

# # [3] Conduct editing using StyleCLIP
cd StyleCLIP
bash scripts/edit.sh $CUDA_DEVICE $EDIT_TEXT $OUTPUT_DIR  
cd ..

# # [4] Extract segmentation map from edited image for FENeRF inversion (editing).
cd FENeRF
bash scripts/edited_segmentation.sh $CUDA_DEVICE $EDIT_TEXT $OUTPUT_DIR
cd ..

# # [5] Conduct FENeRF Inversion
cd FENeRF
if [ "$USE_ORIGINAL_METHOD" == true ]; 
then
    bash scripts/run_editing_inversion.sh $CUDA_DEVICE $FENERF_GENERATOR_PATH $OUTPUT_DIR/RGB_orig $OUTPUT_DIR/${EDIT_TEXT}/segmaps $OUTPUT_DIR/${EDIT_TEXT}/fenerf_final $OUTPUT_DIR/fenerf_latents $OUTPUT_DIR/face_angles_${N_SAMPLES}.pkl
else
    bash scripts/run_editing_inversion_beta.sh $CUDA_DEVICE $FENERF_GENERATOR_PATH $OUTPUT_DIR/${EDIT_TEXT}/edited $OUTPUT_DIR/${EDIT_TEXT}/segmaps $OUTPUT_DIR/${EDIT_TEXT}/fenerf_final_beta $OUTPUT_DIR/fenerf_latents $OUTPUT_DIR/face_angles_${N_SAMPLES}.pkl
fi