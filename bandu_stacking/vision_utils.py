from __future__ import print_function

import os
from collections import OrderedDict
from typing import List

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
import trimesh
from roipoly import RoiPoly
from scipy.spatial.distance import cdist
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.cluster import DBSCAN

import bandu_stacking.pb_utils as pbu
from bandu_stacking.realsense_utils import CALIB_DIR


def obj_file_from_mesh(mesh, under=True):
    """Creates a *.obj mesh string :param mesh: tuple of list of vertices and
    list of faces :return: *.obj mesh string."""
    vertices, faces = mesh.vertices, mesh.triangles
    s = "g Mesh\n"  # TODO: string writer
    for v in vertices:
        assert len(v) == 3
        s += "\nv {}".format(" ".join(map(str, v)))
    for f in faces:
        # assert(len(f) == 3) # Not necessarily true
        f = [i + 1 for i in f]  # Assumes mesh is indexed from zero
        s += "\nf {}".format(" ".join(map(str, f)))
        if under:
            s += "\nf {}".format(" ".join(map(str, reversed(f))))
    return s


def pointcloud_to_mesh(pcd_array, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)

    if not pcd.has_normals():
        pcd.estimate_normals()

    alpha = 0.025
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # create the triangular mesh with the vertices and faces from open3d
    pbu.write(save_path, obj_file_from_mesh(mesh, under=True))


def visualize_graph(G):
    """Visualizes a graph using networkx and matplotlib.

    Parameters:
    - G: A networkx Graph object.
    """
    # Position nodes using the spring layout
    pos = nx.spring_layout(G)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=700,
        edge_color="k",
        linewidths=1,
        font_size=15,
        arrows=True,
        arrowsize=20,
    )
    plt.title("Graph Visualization")
    plt.show()


def remove_statistical_outliers(pcd_array, nb_neighbors=25, std_ratio=0.5):
    """Remove statistical outliers from a point cloud.

    Parameters:
    - pcd: Open3D point cloud object.
    - nb_neighbors: Number of neighbors to consider for computing the average distance.
    - std_ratio: Standard deviation ratio; points with a distance larger than this ratio will be removed.

    Returns:
    - Cleaned point cloud after outlier removal.
    - Indices of the inlier points.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    cleaned_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return np.asarray(cleaned_pcd.points)


def merge_touching_pointclouds(pointclouds, distance_threshold=0.02, min_points=250):
    """Merges point clouds that are touching based on a distance threshold
    using DBSCAN.

    Parameters:
    - pointclouds: List of Open3D point cloud objects.
    - distance_threshold: Distance threshold to consider point clouds as touching.

    Returns:
    - List of merged Open3D point cloud objects.
    """
    # Concatenate all points into a single array
    all_points = np.concatenate([np.asarray(pcd) for pcd in pointclouds], axis=0)
    all_points = remove_statistical_outliers(all_points)

    # Use DBSCAN to cluster all points
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(all_points)
    labels = clustering.labels_

    # Group points by their cluster label to form new point clouds
    merged_pointclouds = []
    for label in np.unique(labels):
        # Extract points belonging to the current cluster
        cluster_points = all_points[labels == label]

        # Create a new point cloud object for the cluster
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        points_array = np.asarray(merged_pcd.points)
        if points_array.shape[0] > min_points:
            merged_pointclouds.append(points_array)

    print("Num pointclouds after merging: " + str(len(merged_pointclouds)))
    return merged_pointclouds


def visualize_multiple_pointclouds(pointclouds_np, colors=None):
    """Visualizes multiple point clouds, each with a different color.

    Parameters:
    - pointclouds_np: List of point clouds as numpy arrays of shape (N, 3).
    - colors: Optional. List of colors for each point cloud. If None, random colors are assigned.

    Each point cloud in the list is converted to an Open3D PointCloud object,
    assigned a unique color, and visualized together.
    """
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for i, pc_np in enumerate(pointclouds_np):
        # Convert numpy array to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)

        # Assign color
        if colors is None:
            # Generate a random color if none is provided
            color = np.random.rand(3)
        else:
            color = colors[i]
        pcd.paint_uniform_color(color)  # Set the color for the point cloud

        # Add the point cloud to the visualization
        vis.add_geometry(pcd)

    # Run the visualization
    vis.run()
    vis.destroy_window()


def depth_mask_to_point_clouds(camera_image: pbu.CameraImage, masks):
    """Convert a depth image to point clouds for each mask.

    Parameters:
    - depth_image: numpy array (HxWx1), the depth image.
    - masks: List of numpy arrays (HxWx1), each representing a binary mask for an object.
    - camera_pose: Tuple containing camera pose (Euler angles, Quaternion).
    - camera_intrinsics: Dictionary containing camera intrinsic parameters
                         such as 'fx', 'fy', 'cx', 'cy'.

    Returns:
    - List of Open3D point cloud objects, each corresponding to a mask.
    """
    # Create Open3D depth image from numpy array
    depth_o3d = o3d.geometry.Image(camera_image.depthPixels.astype(np.float32))

    # Create Open3D intrinsic object
    camera_matrix = camera_image.camera_matrix
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=camera_image.depthPixels.shape[1],
        height=camera_image.depthPixels.shape[0],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    # Apply transformation based on camera pose
    R = pbu.tform_from_pose(camera_image.camera_pose)

    # Create a point cloud from the depth image
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics)
    pcd.transform(R)
    table_inlier_thresh = 0.01
    plane_model, _ = pcd.segment_plane(
        distance_threshold=table_inlier_thresh, ransac_n=3, num_iterations=1000
    )

    # Apply masks and extract individual point clouds
    mask_pointclouds = []
    for mask in masks:

        mask_bool = mask["segmentation"].astype(bool)
        masked_depth = np.where(mask_bool, depth_o3d, 0)
        depth_image_o3d = o3d.geometry.Image(masked_depth.astype(np.float32))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image_o3d, intrinsics
        )
        pcd.transform(R)

        pcd_points = np.asarray(pcd.points)

        # Reject adding this cluster if there are not enough points
        min_cluster_points = 50
        if pcd_points.shape[0] < min_cluster_points:
            continue

        # If a large percent of the cluster are inliers on the table plane, we reject the cluster
        A, B, C, D = plane_model
        table_inlier_ratio = 0.8
        distances = np.abs(
            A * pcd_points[:, 0] + B * pcd_points[:, 1] + C * pcd_points[:, 2] + D
        ) / np.sqrt(A**2 + B**2 + C**2)
        inliers = distances < table_inlier_thresh
        if np.mean(inliers) >= table_inlier_ratio:
            continue

        mask_pointclouds.append(pcd_points)

    # visualize_multiple_pointclouds(mask_pointclouds)
    print("Num pointclouds before merging: " + str(len(mask_pointclouds)))

    return mask_pointclouds


def save_camera_images(
    camera_image: pbu.CameraImage, directory="./logs", prefix="", **kwargs
):
    pbu.ensure_dir(directory)
    pbu.save_image(
        os.path.join(directory, "{}rgb.png".format(prefix)), camera_image.rgbPixels
    )  # [0, 255]

    depth_image = camera_image.depthPixels
    depth_image = (
        (depth_image - np.min(depth_image))
        / (np.max(depth_image) - np.min(depth_image))
        * 255
    ).astype(np.uint8)
    pbu.save_image(
        os.path.join(directory, "{}depth.png".format(prefix)), depth_image
    )  # [0, 1]

    if camera_image.segmentationMaskBuffer is None:
        return

    pbu.save_image(
        os.path.join(directory, "{}segmented.png".format(prefix)),
        camera_image.segmentationMaskBuffer,
    )  # [0, 255]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img


def plot_segmentation(img):
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    ax.imshow(img)
    plt.axis("off")
    plt.show()


def mask_roi(camera_sn, camera_image):

    mask_path = os.path.join(CALIB_DIR, f"{camera_sn}/mask.npy")
    if os.path.exists(mask_path):
        poly_verts = np.load(mask_path)
    else:
        fig = plt.figure()
        plt.imshow(camera_image.rgbPixels, interpolation="nearest", cmap="Greys")
        plt.show(block=False)

        # Let user draw first ROI
        roi1 = RoiPoly(color="r", fig=fig)

        poly_verts = np.array(
            [[(roi1.x[0], roi1.y[0])] + list(zip(reversed(roi1.x), reversed(roi1.y)))]
        )[0, :, :]

        np.save(mask_path, poly_verts)

    polygon_path = mpath.Path(poly_verts)
    y, x = np.indices(camera_image.depthPixels.shape)

    # Flatten the coordinate grid and create pairs of (x, y)
    points = np.vstack((x.flatten(), y.flatten())).T

    inside_polygon = polygon_path.contains_points(points)
    mask = inside_polygon.reshape(camera_image.depthPixels.shape)
    camera_image.rgbPixels[~mask] = 0
    camera_image.depthPixels[~mask] = 0
    return camera_image


def load_sam():

    checkpoint_path = os.path.abspath(
        os.path.join(__file__, *[os.pardir] * 2, "checkpoints")
    )
    sam_checkpoint = os.path.join(checkpoint_path, "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam


def get_seg_sam(sam, image):

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    img = show_anns(masks)
    # plot_segmentation(img)

    return masks, img
