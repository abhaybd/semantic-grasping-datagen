gantry run --budget ai2/prior -w ai2/abhayd \
    --dataset abhayd/obsgen_0311_1420:/data \
    --gpus 1 \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    -- \
    python semantic_grasping_datagen/datagen/collate_data.py /data /results --batch_size 32
