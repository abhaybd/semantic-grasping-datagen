gantry run --budget ai2/prior -w ai2/abhayd \
    --dataset abhayd/acronym_processed:/assets \
    --dataset abhayd/synthetic_annotations_0401_1426:/annotations \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_ACCESS_KEY \
    --gpus 1 \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    --priority normal \
    --cluster ai2/augusta-google-1 \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/prior-elanding \
    --allow-dirty \
    -- \
    /bin/bash scripts/datagen_pipeline.sh
