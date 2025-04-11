gantry run --budget ai2/prior -w ai2/abhayd \
    --weka prior-default:/data \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_ACCESS_KEY \
    --gpus 1 \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    --priority high \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --allow-dirty \
    -- \
    /bin/bash scripts/datagen_pipeline.sh
