DATASET_NAME="0410_2243"
NAME="vlm_datapackage_${DATASET_NAME}"

gantry run --workspace ai2/abhayd --budget ai2/prior \
    --name ${NAME} \
    --task-name ${NAME} \
    --weka prior-default:/data \
    --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    --priority high \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/saturn-cirrascale \
    --allow-dirty \
    -- \
    python semantic_grasping_datagen/create_vlm_data.py \
        /data/abhayd/semantic-grasping-datasets/${DATASET_NAME}/robotpoint_data \
        /data/abhayd/semantic-grasping-datasets/${DATASET_NAME}/observations \
        /data/abhayd/semantic-grasping-datasets/${DATASET_NAME}/dataset.csv \
        --format robopoint
