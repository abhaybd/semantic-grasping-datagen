gantry run --budget ai2/prior -w ai2/abhayd \
    --dataset abhayd/datagen_0310:/data \
    --gpus 1 \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    -- \
    python semantic_grasping_datagen/datagen/generate_obs.py
