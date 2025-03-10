gantry run -w ai2/abhayd -b ai2/prior \
    --dataset abhayd/acronym_processed:/assets \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_ACCESS_KEY \
    --beaker-image ai2/cuda11.8-ubuntu20.04 \
    --no-python \
    --allow-dirty \
    -- \
    aws s3 sync /assets s3://prior-datasets/semantic-grasping/acronym
