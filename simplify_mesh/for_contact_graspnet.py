import numpy as np
import trimesh

# Read in a mesh and convert it into a pointcloud
mesh = trimesh.load(
    "/home/krishna/code/bandu_planning/bandu_stacking/models/bandu_simplified/Barrell_decimated.obj"
)

# Convert the mesh into a pointcloud
pointcloud = mesh.sample(100000)

# Write out a npy file with a single key 'xyz'
# Create a dictionary with a single key 'xyz'
data = {"xyz": pointcloud}
# Save the dictionary to a .npy file
output_path = "/home/krishna/code/bandu_planning/bandu_stacking/models/bandu_simplified/Barrell_decimated_pointcloud.npy"
np.save(output_path, data)
