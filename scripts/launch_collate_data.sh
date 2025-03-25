OBS_DATASET_NAME=obsgen_0317_1735

gantry run --budget ai2/prior -w ai2/abhayd \
    --dataset abhayd/${OBS_DATASET_NAME}:/data \
    --gpus 1 \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    --priority normal \
    --cluster ai2/augusta-google-1 \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/prior-elanding \
    -- \
    python semantic_grasping_datagen/datagen/collate_data.py /data/observations /results --annot-type full
