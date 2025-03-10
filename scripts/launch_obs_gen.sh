if [ $# -ne 1 ]; then
    echo "Usage: $0 DATASET_NAME"
    exit 1
fi

DATASET_NAME=$1

gantry run --dataset abhayd/$DATASET_NAME:/data --budget ai2/prior -w ai2/abhayd --gpus 1 -- \
    python semantic_grasping_datagen/datagen/generate_obs.py
