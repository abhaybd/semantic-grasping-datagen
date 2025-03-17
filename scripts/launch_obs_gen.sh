gantry run --budget ai2/prior -w ai2/abhayd \
    --dataset abhayd/datagen_0310_1527:/data \
    --gpus 1 \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    --priority normal \
    --cluster ai2/prior-elanding \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    -- \
    python semantic_grasping_datagen/datagen/generate_obs.py
