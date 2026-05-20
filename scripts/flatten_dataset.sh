#!/bin/bash
/* Flattens a dataset by copying all files from subdirectories to a single directory.
    Usage: ./flatten_dataset.sh <root_directory> <output_directory>
    
*/

ROOT=$1
OUT_DIR=$2

if [[ ! -d "$OUT_DIR" ]]; then
    mkdir -p "$OUT_DIR"
fi

for folder in "$ROOT"/*; do
    if [[ -d "$folder" ]]; then
        cp "$folder"/* "$OUT_DIR"/
    fi
done