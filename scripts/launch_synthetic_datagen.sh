gantry run --workspace ai2/abhayd --budget ai2/prior \
    --dataset abhayd/acronym_processed:/assets \
    --env-secret OPENAI_API_KEY=OPENAI_API_KEY \
    --gpus 1 \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    --priority normal \
    --cluster ai2/prior-elanding \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/augusta-google-1 \
    -- \
    python semantic_grasping_datagen/synthetic_annotations.py submit \
        categories.txt \
        /assets \
        --collage_size 2 2 \
        --resolution 640 480 \
        --blacklist_file asset_blacklist.txt \
        --out_dir /results
