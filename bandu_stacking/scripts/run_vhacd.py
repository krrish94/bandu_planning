from __future__ import annotations

import os
import shutil

import pybullet as p

if __name__ == "__main__":
    directory = "../models/bandu_simplified"
    for filename in os.listdir(directory):
        if ".obj" in filename and "convex" in filename:

            path = os.path.join(directory, filename)
            # shutil.copy(path, path.replace(".obj", "_convex.obj"))

            print("Running VHACD on " + path)

            in_obj_file = path
            out_obj_file = path.replace("_convex", "")
            log_file = path + "_vhacd_log.txt"

            # Set up the options
            options = {
                "resolution": 100000,
                "depth": 20,
                "concavity": 0.0025,
                "planeDownsampling": 4,
                "convexhullDownsampling": 4,
                "alpha": 0.05,
                "beta": 0.05,
                "gamma": 0.0005,
                "pca": 0,
                "mode": 0,
                "maxNumVerticesPerCH": 64,
                "minVolumePerCH": 0.0001,
            }

            # Perform the decomposition
            p.vhacd(in_obj_file, out_obj_file, log_file, **options)
