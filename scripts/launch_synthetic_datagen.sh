DATASET_NAME=0409_2242

gantry run --workspace ai2/abhayd --budget ai2/prior \
    --weka prior-default:/data \
    --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
    --gpus 1 \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    --priority high \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/saturn-cirrascale \
    --allow-dirty \
    -- \
    python semantic_grasping_datagen/synthetic_annotations.py submit \
        all_categories.txt \
        /data/abhayd/semantic-grasping-datasets/acronym_processed \
        --collage_size 2 2 \
        --resolution 640 480 \
        --blacklist_file asset_blacklist.txt \
        --batch-ids-file /results/batch_ids.txt \
        --out_dir /data/abhayd/semantic-grasping-datasets/${DATASET_NAME}/annotations
