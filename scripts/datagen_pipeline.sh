python semantic_grasping_datagen/datagen/datagen.py out_dir=results/scenes  # write to local results folder instead of bucket
python semantic_grasping_datagen/datagen/generate_obs.py scene_dir=results/scenes out_dir=/results/observations
python semantic_grasping_datagen/datagen/collate_data.py /results/observations /results --annot-type full
