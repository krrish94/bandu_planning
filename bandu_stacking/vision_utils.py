from __future__ import print_function

import os
from collections import OrderedDict

import numpy as np

import bandu_stacking.pb_utils as pbu
import matplotlib.pyplot as plt
from roipoly import RoiPoly
import matplotlib.path as mpath
from bandu_stacking.realsense_utils import CALIB_DIR
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import open3d as o3d
from typing import List
from scipy.spatial.distance import cdist


UNKNOWN = "unknown"
TABLE = "table"
SPECIAL_CATEGORIES = {None: pbu.BLACK, UNKNOWN: pbu.GREY, TABLE: pbu.WHITE}



def merge_touching_pointclouds(pointclouds, distance_threshold=0.05):
    """
    Merges point clouds that are touching based on a distance threshold.

    Parameters:
    - pointclouds: List of Open3D point cloud objects.
    - distance_threshold: Distance threshold to consider point clouds as touching.

    Returns:
    - List of merged Open3D point cloud objects.
    """

    def are_touching(pcd1, pcd2, threshold):
        """Check if two point clouds are touching based on the threshold."""
        # Compute the minimum distance between any two points in the point clouds
        dists = cdist(np.asarray(pcd1.points), np.asarray(pcd2.points), 'euclidean')
        min_dist = np.min(dists)
        return min_dist < threshold

    # Create a graph where an edge represents that point clouds are touching
    num_pcds = len(pointclouds)
    edges = []
    for i in range(num_pcds):
        for j in range(i + 1, num_pcds):
            if are_touching(pointclouds[i], pointclouds[j], distance_threshold):
                edges.append((i, j))

    # Find connected components in the graph
    from networkx import Graph, connected_components
    G = Graph()
    G.add_edges_from(edges)
    components = list(connected_components(G))

    # Merge point clouds in each connected component
    merged_pointclouds = []
    for component in components:
        merged_pcd = o3d.geometry.PointCloud()
        for index in component:
            merged_pcd += pointclouds[index]
        merged_pointclouds.append(merged_pcd)

    return merged_pointclouds

def depth_mask_to_point_clouds(camera_image:pbu.CameraImage, masks):
    """
    Convert a depth image to point clouds for each mask.

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
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width=camera_image.depthPixels.shape[1],
                                                   height=camera_image.depthPixels.shape[0],
                                                   fx=fx, fy=fy, cx=cx, cy=cy)

    # Create a point cloud from the depth image
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics)
    table_inlier_thresh = 0.01
    plane_model, _ = pcd.segment_plane(distance_threshold=table_inlier_thresh,
                                             ransac_n=3,
                                             num_iterations=1000)
        
    # Apply transformation based on camera pose
    # Assuming camera_pose is (euler angles, quaternion)
    euler, quaternion = camera_image.camera_pose
    if quaternion:  # If quaternion is provided
        R = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    else:  # Convert euler angles to rotation matrix
        R = o3d.geometry.get_rotation_matrix_from_xyz(euler)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    pcd.transform(transformation_matrix)

    # Apply masks and extract individual point clouds
    mask_pointclouds = []
    for mask in masks:
        # Convert mask to boolean array
        mask_bool = mask.astype(bool)

        # Extract points corresponding to the mask
        points = np.asarray(pcd.points)[mask_bool.flatten(), :]
        mask_pcd = o3d.geometry.PointCloud()
        mask_pcd.points = o3d.utility.Vector3dVector(points)
        

        # Reject adding this cluster if there are not enough points
        min_cluster_points = 50
        if(points.shape[0]<min_cluster_points):
            continue

        # If a large percent of the cluster are inliers on the table plane, we reject the cluster
        A, B, C, D = plane_model
        table_inlier_ratio = 0.8
        distances = np.abs(A*points[:, 0] + B*points[:, 1] + C*points[:, 2] + D) / np.sqrt(A**2 + B**2 + C**2)
        inliers = distances < table_inlier_thresh
        if np.mean(inliers) >= table_inlier_ratio:
            continue
            
        mask_pointclouds.append(mask_pcd)

    print("Num clusters: "+str(len(mask_pointclouds)))

    return mask_pointclouds

def image_from_labeled(seg_image, **kwargs):

    # TODO: order special colors
    # TODO: adjust saturation and value per category
    # labels = sorted(set(get_bodies()) | set(seg_image[..., 0].flatten()))
    labels_instance = set(seg_image[..., 1].flatten())
    detect_obj_labels = sorted(
        label
        for label in labels_instance
        if (label not in SPECIAL_CATEGORIES)
    )
    labels = detect_obj_labels
    color_from_body = OrderedDict(zip(labels, pbu.spaced_colors(len(labels))))
    color_from_body.update(SPECIAL_CATEGORIES)

    image = np.zeros(seg_image.shape[:2] + (3,))
    for r in range(seg_image.shape[0]):
        for c in range(seg_image.shape[1]):
            category, instance = seg_image[r, c, :]
            if category in color_from_body:  # SPECIAL_CATEGORIES:
                color = color_from_body[category]
            else:
                color = color_from_body[instance]
            image[r, c, :] = color[:3]
    return (image * 255).astype(np.uint8)


def save_camera_images(
    camera_image: pbu.CameraImage, directory="./logs", prefix="", **kwargs
):
    pbu.ensure_dir(directory)
    pbu.save_image(
        os.path.join(directory, "{}rgb.png".format(prefix)), camera_image.rgbPixels
    )  # [0, 255]

    depth_image = camera_image.depthPixels
    depth_image = ((depth_image-np.min(depth_image))/(np.max(depth_image)-np.min(depth_image))*255).astype(np.uint8)
    pbu.save_image(
        os.path.join(directory, "{}depth.png".format(prefix)), depth_image
    )  # [0, 1]

    if camera_image.segmentationMaskBuffer is None:
        return None

    segmented_image = image_from_labeled(camera_image.segmentationMaskBuffer, **kwargs)
    print(segmented_image)
    pbu.save_image(
        os.path.join(directory, "{}segmented.png".format(prefix)), segmented_image
    )  # [0, 255]
    return segmented_image

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def mask_roi(camera_sn, camera_image):

    mask_path = os.path.join(CALIB_DIR, f"{camera_sn}/mask.npy")
    if(os.path.exists(mask_path)):
        poly_verts = np.load(mask_path)
    else:
        fig = plt.figure()
        plt.imshow(camera_image.rgbPixels, interpolation='nearest', cmap="Greys")
        plt.show(block=False)

        # Let user draw first ROI
        roi1 = RoiPoly(color='r', fig=fig)

        poly_verts = np.array([[(roi1.x[0], roi1.y[0])]
                + list(zip(reversed(roi1.x), reversed(roi1.y)))])[0, :, :]
        
        np.save(mask_path, poly_verts)
    
    polygon_path = mpath.Path(poly_verts)
    y, x = np.indices(camera_image.depthPixels.shape)

    # Flatten the coordinate grid and create pairs of (x, y)
    points = np.vstack((x.flatten(), y.flatten())).T

    inside_polygon = polygon_path.contains_points(points)
    mask = inside_polygon.reshape(camera_image.depthPixels.shape)
    camera_image.rgbPixels[~mask] = 0
    return camera_image

def load_sam():

    checkpoint_path = os.path.abspath(os.path.join(__file__, *[os.pardir] * 2, "checkpoints"))
    sam_checkpoint = os.path.join(checkpoint_path, "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam


def get_seg_sam(sam, image):
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

    return masks
