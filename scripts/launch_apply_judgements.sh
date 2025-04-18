if [ $# -ne 1 ]; then
    echo "Usage: $0 <data_id>"
    exit 1
fi

DATA_ID=$1
OUT_DIR="/weka/prior/abhayd/semantic-grasping-datasets/synthetic_annotations_filtered_${DATA_ID}"

gantry run --workspace ai2/abhayd --budget ai2/prior \
    --name apply_judgements_${DATA_ID} \
    --task-name apply_judgements_${DATA_ID} \
    --weka prior-default:/weka/prior \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_ACCESS_KEY \
    --priority high \
    --cluster ai2/ceres-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/saturn-cirrascale \
    --allow-dirty \
    -- \
    python semantic_grasping_datagen/apply_judgements.py ${OUT_DIR}
