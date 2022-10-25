#!/bin/bash
CUDA_DEVICE=${1:-0}
EDIT_TEXT=${2:-"curly_hair"}
GEN_ORIG=${3:-false}
USE_ORIGINAL_METHOD=${4:-false}


EDIT_TEXTS=("elf_ear" "open_mouth" "big_lips")

for i in "${!EDIT_TEXTS[@]}"; do
    bash run_jhyung.sh ${CUDA_DEVICE} "${EDIT_TEXTS[i]}" false false
done