# Semantic Grasping Datagen

This repository contains code for data generation, annotation, and inspection.

`data_annotation` contains a frontend for data annotation, served by the server in `src/annotation_server.py`. This is dockerized by `Dockerfile`.

The scripts in `scripts` launch beaker jobs for handling the data preprocessing and generation pipeline.

Note that in trimesh, the camera frame is defined by +x=right, +y=up, +z=backwards, whereas usually the camera frame is +x=right, +y=down, +z=forward.
The camera frames stored in the datageneration pipeline are in the trimesh conventions, but since backprojection yields a pointcloud in the standard conventions,
the grasps will be stored in the standard camera frame, not the trimesh one.
