#!/bin/bash
CUDA_DEVICE=${1:-0}
EDIT_TEXT=${2:-"curly_hair"}
GEN_ORIG=${3:-false}
USE_ORIGINAL_METHOD=${4:-false}

#EDIT_TEXTS=("arched_eyebrows" "bald" "bangs" \
#            "big_lips" "big_nose" "blue_nose" \
#            "brown_hair" "bushy_eyebrows" "closed_eyes" \
#            "elf_ear" "eyeglasses" "goatee"\
#            "green_lips" "grey_hair" "happy" "mustache" "open_mouth" \
#            "pale_skin" "purple_nose" "sad" "smiling" "straight_hair" "yellow_lips")

#EDIT_TEXTS=("arched_eyebrows" "bald" "bangs" \
#            "big_lips" "big_nose" \
#            "brown_hair" "bushy_eyebrows" "closed_eyes" \
#            "elf_ear" "eyeglasses" "goatee"\
#            "grey_hair" "happy" "mustache" "open_mouth" \
#            "pale_skin" "sad" "smiling" "straight_hair" "red_lips")
EDIT_TEXTS=("bangs" "arched_eyebrows" "bushy_eyebrows")

for i in "${!EDIT_TEXTS[@]}"; do
    bash run_jhyung.sh ${CUDA_DEVICE} "${EDIT_TEXTS[i]}" false false
done