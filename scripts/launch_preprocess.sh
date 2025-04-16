gantry run --budget ai2/prior -w ai2/abhayd \
    --weka prior-default:/data \
    --priority high \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --allow-dirty \
    -- \
    python semantic_grasping_datagen/preprocess_shapenet.py \
        /data/abhayd/semantic-grasping-datasets/acronym/grasps \
        /data/abhayd/semantic-grasping-datasets/acronym/ShapeNetSem \
        /data/abhayd/semantic-grasping-datasets/acronym_processed \
        --blacklist asset_blacklist.txt \
        --sampling-categories-file all_categories.txt \
        --step project
