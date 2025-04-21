#!/bin/bash

set -euxo pipefail

DATASET_PATH=/data/abhayd/semantic-grasping-datasets/0417_2121
ASSETS_PATH=/data/abhayd/semantic-grasping-datasets/acronym_processed
ANNOTS_PATH=/data/abhayd/semantic-grasping-datasets/synthetic_annotations_filtered_0417_2015
TASKS_JSON=/data/abhayd/semantic-grasping-datasets/semantic_task_cleaned_up_implicit.json
FORMAT=molmo

python semantic_grasping_datagen/datagen/datagen.py \
    out_dir=${DATASET_PATH}/scenes \
    data_dir=${ASSETS_PATH} \
    n_samples=10000 \
    "annotation_sources=[{type:directory,params:{dir:${ANNOTS_PATH}}}]"
python semantic_grasping_datagen/datagen/generate_obs.py scene_dir=${DATASET_PATH}/scenes out_dir=${DATASET_PATH}/observations
python semantic_grasping_datagen/datagen/collate_data.py ${DATASET_PATH}/observations ${DATASET_PATH} --annot-type full
python semantic_grasping_datagen/datagen/match_tasks_to_grasps.py \
    ${TASKS_JSON} \
    ${DATASET_PATH}/observations \
    ${DATASET_PATH}/task_point \
    --n-proc 32
python semantic_grasping_datagen/datagen/package_pointing_data.py \
    ${DATASET_PATH}/task_point/matched_tasks.csv \
    ${DATASET_PATH}/observations \
    ${DATASET_PATH}/${FORMAT}_data \
    --format ${FORMAT} \
    --n-proc 32
