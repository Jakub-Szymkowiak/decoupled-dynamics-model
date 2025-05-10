#!/bin/bash

SCENE_NAME="$1"
shift

if [ -z "$SCENE_NAME" ]; then
    echo "Usage: $0 <scene-name> [additional train.py arguments]"
    exit 1
fi

SCENE_ROOT="/home/computergraphics/Documents/jszymkowiak/project/decoupled-dynamics-model/data"
SCENE_PATH="${SCENE_ROOT}/${SCENE_NAME}"
OUTPUT_DIR="./output/${SCENE_NAME}/_001"

mkdir -p "${OUTPUT_DIR}"

python train.py -s "${SCENE_PATH}" -m "${OUTPUT_DIR}" "$@"