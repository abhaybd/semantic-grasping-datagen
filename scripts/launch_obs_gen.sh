if [ $# -ne 1 ]; then
    echo "Usage: $0 DATASET_NAME"
    exit 1
fi

DATASET_NAME=$1

gantry run --budget ai2/prior -w ai2/abhayd \
    --dataset abhayd/$DATASET_NAME:/data \
    --gpus 1 \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    -- \
    python semantic_grasping_datagen/datagen/generate_obs.py
