gantry run --workspace ai2/abhayd --budget ai2/prior \
    --dataset abhayd/acronym_processed:/data \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_ACCESS_KEY \
    -- \
    python semantic_grasping_datagen/datagen/datagen.py
