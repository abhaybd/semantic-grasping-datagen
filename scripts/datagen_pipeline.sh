DATASET_PATH=/data/abhayd/semantic-grasping-datasets/0407_1606
ASSETS_PATH=/data/abhayd/semantic-grasping-datasets/acronym_processed
python semantic_grasping_datagen/datagen/datagen.py out_dir=${DATASET_PATH}/scenes data_dir=${ASSETS_PATH}
python semantic_grasping_datagen/datagen/generate_obs.py scene_dir=${DATASET_PATH}/scenes out_dir=${DATASET_PATH}/observations
python semantic_grasping_datagen/datagen/collate_data.py ${DATASET_PATH}/observations ${DATASET_PATH} --annot-type full
