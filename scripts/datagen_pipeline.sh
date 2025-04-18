#!/bin/bash

set -euxo pipefail

DATASET_PATH=/data/abhayd/semantic-grasping-datasets/0417_2121
ASSETS_PATH=/data/abhayd/semantic-grasping-datasets/acronym_processed
ANNOTS_PATH=/data/abhayd/semantic-grasping-datasets/synthetic_annotations_filtered_0417_2015
VLM_FORMAT="molmo"

python semantic_grasping_datagen/datagen/datagen.py \
    out_dir=${DATASET_PATH}/scenes \
    data_dir=${ASSETS_PATH} \
    n_samples=10000 \
    "annotation_sources=[{type:directory,params:{dir:${ANNOTS_PATH}}}]"
python semantic_grasping_datagen/datagen/generate_obs.py scene_dir=${DATASET_PATH}/scenes out_dir=${DATASET_PATH}/observations
python semantic_grasping_datagen/datagen/collate_data.py ${DATASET_PATH}/observations ${DATASET_PATH} --annot-type full
python semantic_grasping_datagen/create_vlm_data.py \
    ${DATASET_PATH}/${VLM_FORMAT}_data \
    ${DATASET_PATH}/observations \
    ${DATASET_PATH}/dataset.csv \
    --format ${VLM_FORMAT} \
    --n-proc 32
