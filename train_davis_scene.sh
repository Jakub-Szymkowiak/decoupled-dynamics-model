#!/bin/bash

SCENE_NAME="$1"
ID="$2"

if [ -z "$SCENE_NAME" ]; then
    echo "Usage: $0 <scene-name> <id> [additional train.py arguments]"
    exit 1
fi

shift 2

SCENE_ROOT="./data"
SCENE_PATH="${SCENE_ROOT}/${SCENE_NAME}"
OUTPUT_DIR="./output/${SCENE_NAME}/_${ID}"

mkdir -p "${OUTPUT_DIR}"

python train.py -s "${SCENE_PATH}" -m "${OUTPUT_DIR}" "$@"
