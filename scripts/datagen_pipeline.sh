#!/bin/bash

set -euxo pipefail

DATASET_PATH=/weka/prior/abhayd/semantic-grasping-datasets/0417_2121
ASSETS_PATH=/weka/prior/abhayd/semantic-grasping-datasets/acronym_processed
ANNOTS_PATH=/weka/prior/abhayd/semantic-grasping-datasets/synthetic_annotations_filtered_0417_2015
TASKS_JSON=/weka/prior/abhayd/semantic-grasping-datasets/semantic_task_cleaned_up_implicit.json
FORMAT=molmo
SPLIT=train

python semantic_grasping_datagen/datagen/split_data.py \
    data_dir=${ASSETS_PATH} \
    out_dir=${DATASET_PATH}/splits
python semantic_grasping_datagen/datagen/datagen.py \
    out_dir=${DATASET_PATH}/scenes \
    data_dir=${ASSETS_PATH} \
    split_file=${DATASET_PATH}/splits/${SPLIT}.json \
    n_samples=10000 \
    "annotation_sources=[{type:directory,params:{dir:${ANNOTS_PATH}}}]"
python semantic_grasping_datagen/datagen/generate_obs.py scene_dir=${DATASET_PATH}/scenes out_dir=${DATASET_PATH}/observations
# python semantic_grasping_datagen/datagen/collate_data.py ${DATASET_PATH}/observations ${DATASET_PATH} --annot-type full
python semantic_grasping_datagen/datagen/match_tasks_to_grasps_v2.py \
    ${TASKS_JSON} \
    ${DATASET_PATH}/observations \
    ${DATASET_PATH}/task_point_v2 \
    --n-proc 32 \
    --submit \
    --retrieve
python semantic_grasping_datagen/datagen/package_pointing_data.py \
    ${DATASET_PATH}/task_point_v2/matched_tasks.csv \
    ${DATASET_PATH}/observations \
    ${DATASET_PATH}/${FORMAT}_data \
    --format ${FORMAT} \
    --cot \
    --n-proc 32
