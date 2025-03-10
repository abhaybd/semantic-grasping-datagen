gantry run --dataset abhayd/acronym:/acronym --budget ai2/prior -w ai2/abhayd -- \
    python semantic_grasping_datagen/preprocess_shapenet.py /acronym/grasps /acronym/ShapeNetSem /results \
    --blacklist asset_blacklist.txt --sampling-categories-file all_categories.txt
