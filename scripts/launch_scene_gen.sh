gantry run --workspace ai2/abhayd --budget ai2/prior \
    --dataset abhayd/acronym_processed:/assets \
    --dataset abhayd/synthetic_annotations_0401_1426:/annotations \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_ACCESS_KEY \
    --priority normal \
    --cluster ai2/augusta-google-1 \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/prior-elanding \
    -- \
    python semantic_grasping_datagen/datagen/datagen.py
