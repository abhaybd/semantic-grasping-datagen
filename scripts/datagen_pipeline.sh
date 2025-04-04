python semantic_grasping_datagen/datagen/datagen.py out_dir=/results/scenes
python semantic_grasping_datagen/datagen/generate_obs.py scene_dir=/results/scenes out_dir=/results/observations
python semantic_grasping_datagen/datagen/collate_data.py /results/observations /results --annot-type full
rm -r /results/scenes  # delete original scenes to save space
