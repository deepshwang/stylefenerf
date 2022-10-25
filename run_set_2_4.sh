#!/bin/bash
CUDA_DEVICE=${1:-0}

#EDIT_TEXTS=("arched_eyebrows" "bald" "bangs" "big_lips" "big_nose" "blue_nose" \
#            "brown_hair" "bushy_eyebrows" "closed_eyes" "elf_ear" "eyeglasses" \
#            "goatee" "green_lips" "grey_hair" "happy" "mustache" "open_mouth" \
#            "pale_skin" "purple_nose" "sad" "smiling" "straight_hair" "yellow_lips")

EDIT_TEXTS=("brown_hair" "bushy_eyebrows" "closed_eyes" "elf_ear" "eyeglasses")
for i in "${!EDIT_TEXTS[@]}"; do
    bash run.sh ${CUDA_DEVICE} "${EDIT_TEXTS[i]}" false false
done