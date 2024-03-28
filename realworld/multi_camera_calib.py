import copy
import os
import numpy as np
import cv2
import json
import pickle as pkl

import open3d as o3d
import pyrealsense2 as rs
from PIL import Image
from tqdm import trange
from realsense_recorder import realsense_capture, get_profiles
import argparse
import random
import glob
import apriltag

def capture_realsense_image(output_folder, serial_number):

    from types import SimpleNamespace
    rs_args = SimpleNamespace()
    rs_args.output_folder = output_folder
    rs_args.realsense_preset = 1
    rs_args.clipping_distance = 3
    rs_args.frames_to_capture = 1
    rs_args.render_images = False
    rs_args.record_images = True
    rs_args.color_profile = 14  #42
    rs_args.depth_profile = 5

    os.makedirs(output_folder, exist_ok=True)

    # Create a pipeline -- use Open3D's implementation
    pipeline = rs.pipeline()
    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()
    config = rs.config()
    config.enable_device(serial_number)
    profile = config.resolve(pipeline)
    # print(profile)
    # quit()
    color_profiles, depth_profiles = get_profiles()
    # for _profile_to_print in color_profiles:
    #     print(_profile_to_print)

    # note: using 640 x 480 depth resolution produces smooth depth boundaries for manipulator experiments
    # using rs.format.rgb8 for color image format for OpenCV based image visualization (to visualize properly --> convert color formatting scheme accordingly)
    color_profile = color_profiles[rs_args.color_profile]
    depth_profile = depth_profiles[rs_args.depth_profile]

    print(f'Using the profiles: color: {color_profile}, depth: {depth_profile}')
    # w, h, fps, fmt = depth_profile
    # config.enable_stream(rs.stream.depth, w, h, fmt, fps)
    # w, h, fps, fmt = color_profile
    # config.enable_stream(rs.stream.color, w, h, fmt, fps)
    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    realsense_capture(rs_args, pipeline, depth_sensor, align, serial_number, frames_to_skip=10)


def read_pcds(imgdir, serial_numbers, depth_cutoffs):
    
    pcds = []
    for idx, serial_number in enumerate(serial_numbers):
        rgb_image_path = os.path.join(imgdir, serial_number, "rgb", "00000.jpg")
        rgb_img = o3d.io.read_image(rgb_image_path)
        depth_image_path = os.path.join(imgdir, serial_number, "depth", "00000.png")
        depth_img_np = np.array(Image.open(depth_image_path), dtype=np.uint16)
        clipping_distance_in_meters = depth_cutoffs[idx]  # in meters
        depth_img_np = depth_img_np.astype(float) / 1000.0
        depth_img_np[depth_img_np > clipping_distance_in_meters] = 0
        depth_img_np = depth_img_np * 1000.0
        depth_img_np = depth_img_np.astype(np.uint16)
        intrinsic_json = os.path.join(imgdir, serial_number, "camera_intrinsic.json")
        with open(intrinsic_json, 'r') as json_file:
            intrinsic_data = json.load(json_file)
        width = intrinsic_data['width']
        height = intrinsic_data['height']
        intrinsic_matrix = intrinsic_data['intrinsic_matrix']
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic_matrix[0], intrinsic_matrix[4], intrinsic_matrix[6], intrinsic_matrix[7]
        )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img, o3d.geometry.Image(depth_img_np), convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic
        )
        # o3d.visualization.draw_geometries([pcd])
        pcds.append(pcd)
    return pcds


def register_icp(pcds, init=None, debug_visualization=False):
    camera_order = [0, 1] # TODO: read the camera order from the config file
    # TODO: Error handling if number of poses in the camera order < 1
    # camera_order = reconstruction_config['realsense_camera_order'] # defines the order in which the poses are processed by ICP --> for good initialization

    registered_pcd = pcds[camera_order[0]]
    registered_pcd.estimate_normals()
    transformations = dict() # transformation matrices representing poses
    transformations[camera_order[0]] = np.eye(4, dtype=np.float32)

    # TODO: Add progress tracker
    for _, camera_pose in enumerate(camera_order[1:]):
        current_pcd = pcds[camera_pose]
        current_pcd.estimate_normals()
        # register using the point2plane estimation method
        # init with identity matrix, if init is not passed
        if init is not None:
            init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            current_pcd, registered_pcd, max_correspondence_distance=0.05, init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())

        # transform the current_pcd by the estimated transformation
        current_pcd = current_pcd.transform(reg_p2p.transformation)
        transformations[camera_pose] = np.asarray(reg_p2p.transformation, dtype=np.float32)
        # merge the pcds to get the registered pcd for the current step
        registered_pcd = merge_pcds([current_pcd, registered_pcd])
        registered_pcd.estimate_normals()
        if debug_visualization: # visualize the registered pcd at the current step if debug_visualization is set to True
            o3d.visualization.draw_geometries([registered_pcd])

    return transformations, registered_pcd

def array2pcd(points, colors):
    """
    Convert points and colors into open3d point cloud.

    Args:
        points(np.array): coordinates of the points.
        colors(np.array): RGB values of the points.

    Returns:
        open3d.geometry.PointCloud: the point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
            # % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])#,
                                    #   zoom=0.4559,
                                    #   front=[0.6452, -0.3036, -0.7011],
                                    #   lookat=[1.9892, 2.0208, 1.8945],
                                    #   up=[-0.2779, -0.9482, 0.1556])


def global_registration(source, target, voxel_size=0.05, visualize=False, fast=False):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    if fast:
        result_ransac = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    else:
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    # print(result_ransac)
    if visualize:
        draw_registration_result(source_down, target_down, result_ransac.transformation)
    return result_ransac


def run_search_for_calibration(source, target, savedir, iters):

    fitnesses = []
    registration_results = []
    inlier_rmses = []

    for i in trange(iters):
        registration_result = global_registration(source, target, voxel_size=0.01, visualize=False)
        # source.transform(registration_result.transformation)
        # print(registration_result.transformation)
        # source.transform(registration_result.transformation)
        if registration_result.fitness == 0 and registration_result.inlier_rmse == 0:
            continue
        registration_results.append(registration_result.transformation)
        fitnesses.append(registration_result.fitness)
        inlier_rmses.append(registration_result.inlier_rmse)
        
        # registration_result = global_registration(source, target, voxel_size=0.01, visualize=True)
        # print(registration_result.transformation)
        # source.transform(registration_result.transformation)

        # registration_result = global_registration(source, target, voxel_size=0.005, visualize=True)
        # print(registration_result.transformation)
        # source.transform(registration_result.transformation)
    
    saved_results = {
        "fitnesses": fitnesses,
        "inlier_rmses": inlier_rmses,
        "registration_results": registration_results,
    }

    with open(os.path.join(savedir, "global_reg_results.pkl"), "wb") as f:
        pkl.dump(saved_results, f)


def pcd2array(pcd):
    """
    Convert open3d point cloud into points and colors.

    Args:
        pcd(open3d.geometry.PointCloud): the point cloud.


    Returns:
        np.array, np.array: coordinates of the points, RGB values of the points.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors


def merge_pcds(pcds):
    """
    Merge several point cloud into a single one.

    Args:
        pcds(list): list of point cloud.

    Returns:
        open3d.geometry.PointCloud: the merged point cloud.
    """
    ret_pcd = o3d.geometry.PointCloud()
    if len(pcds) == 0:
        return ret_pcd
    old_points, old_colors = pcd2array(pcds[0])
    for i in range(1, len(pcds)):
        points, colors = pcd2array(pcds[i])
        old_points = np.concatenate((old_points, points))
        old_colors = np.concatenate((old_colors, colors)) 
    return array2pcd(old_points, old_colors)


def pick_points_from_gui(pcd: o3d.geometry.PointCloud) -> np.array:
    """Return a set of clicked points from the visualizer

    Args:
        pcd (o3d.geometry.PointCloud): Open3D pointcloud to visualize

    Returns:
        _type_: _description_
    """
    print("")
    print("==> Please pick a point using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("==> Afther picking a point, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()



def draw_tag_detections(detector_results, image):
    # loop over the AprilTag detection results
    for r in detector_results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        id = str(r.tag_id)
        cv2.putText(
            image,
            id,
            (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
        # print("[INFO] tag family: {}".format(tagFamily))
    return image


def test_rotation_matrix(R):
    c1 = R[0:3, 0]
    c2 = R[0:3, 1]
    c3 = R[0:3, 2]
    print(np.linalg.norm(c1))
    print(np.linalg.norm(c2))
    print(np.linalg.norm(c3))
    print(np.dot(c1, c2))
    print(np.dot(c2, c3))
    print(np.dot(c3, c1))


def flag_redundant_elements_with_sentinel(l, sentinel=-1):
    _l = []
    for e in l:
        if l.count(e) == 1:
            _l.append(e)
        else:
            _l.append(sentinel)
    return _l


def create_pointcloud(color, depth, intrinsics):
    """https://codereview.stackexchange.com/questions/79032/generating-a-3d-point-cloud"""
    fx, fy, cx, cy = intrinsics
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = depth > 0
    z = np.where(valid, depth / 1000.0, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))


def detection_to_pose(detection, intrinsics, tagsize=None):
    """experimental. did not work well (figured out the bugs later)
    Need to use the scale factor recommended by the homography code (i.e.,
    use the sqrt of the sum of squares of the first two col vectors of the H matrix)
    for scaling the rots and translations. Then, need to scale the resultant
    translation vector by the tagsize (in metres).
    """
    H = detection[5]
    fx, fy, cx, cy = intrinsics

    T = np.zeros((4, 4))
    T[3][3] = 1.0

    T[2][0] = H[2][0]
    T[2][1] = H[2][1]
    T[2][3] = H[2][2]

    T[0][0] = (H[0][0] - cx * T[2][0]) / fx
    T[0][1] = (H[0][1] - cx * T[2][1]) / fx
    T[0][3] = (H[0][2] - cx * T[2][3]) / fx

    T[1][0] = (H[1][0] - cy * T[2][0]) / fy
    T[1][1] = (H[1][1] - cy * T[2][1]) / fy
    T[1][3] = (H[1][2] - cy * T[2][3]) / fy

    if tagsize is None:
        s = 1 / np.sqrt((np.linalg.norm(T[0:3, 0]) + np.linalg.norm(T[0:3, 1])))
    else:
        s = tagsize

    T[0:3, 3] = T[0:3, 3] * s
    T[0:3, 0] = T[0:3, 0]
    T[0:3, 1] = T[0:3, 1]

    T[0][2] = T[1][0] * T[2][1] - T[2][0] * T[1][1]
    T[1][2] = T[2][0] * T[0][1] - T[0][0] * T[2][1]
    T[2][2] = T[0][0] * T[1][1] - T[1][0] * T[0][1]

    col0 = T[0:3, 0]
    col0 = col0 / (np.linalg.norm(col0) + 1e-12)
    T[0:3, 0] = col0
    col1 = T[0:3, 1]
    col1 = col1 / (np.linalg.norm(col1) + 1e-12)
    T[0:3, 1] = col1
    col2 = np.cross(col0, col1)
    col2 = col2 / (np.linalg.norm(col2) + 1e-12)
    T[0:3, 2] = col2

    u, _, v = np.linalg.svd(T[0:3, 0:3])
    T[:3, :3] = np.matmul(u, v)

    return T


def detection_to_metric_pose(detector, detection, intrinsics, tagsize):
    """
    detector: apriltag.Detector object
    detection: apriltag detection result objects
    intrinsics: fx, fy, cx, cy (in units of focal length)
    tagsize: size (side length) of the tag (in metres)
    """
    metric_pose = detector.detection_pose(detection, intrinsics)[0]
    metric_pose[:3, 3] = metric_pose[:3, 3] * tagsize
    return metric_pose


def vector6(x, y, z, a, b, c):
    """Create 6d double numpy array."""
    return np.array([x, y, z, a, b, c], dtype=float)


def get_valid_points_in_both(
    pts1, depth_img_1, pts2, depth_img_2, depth_scale_factor=1000.0
):
    # pts1: 2D tag corners in first image
    # depth_img_1: first depth image
    # pts2: 2D tag corners in second image
    # depth_img_2: second depth image

    # Find points that are common (have valid, non-zero depths) in both images

    rounded_pts_1 = np.round(pts1).astype(np.uint16)
    depths1 = depth_img_1[rounded_pts_1[:, 1], rounded_pts_1[:, 0]]
    depths1 = depths1.astype(np.float32) / depth_scale_factor
    valid_inds_1 = depths1 > 0

    rounded_pts_2 = np.round(pts2).astype(np.uint16)
    depths2 = depth_img_2[rounded_pts_2[:, 1], rounded_pts_2[:, 0]]
    depths2 = depths2.astype(np.float32) / depth_scale_factor
    valid_inds_2 = depths2 > 0

    return np.logical_and(valid_inds_1, valid_inds_2)


def backproject_to_3d(
    pts, depth_img, intrinsics, valid_inds, depth_scale_factor=1000.0
):
    # pts: np.ndarray of shape N x 2 (pts may be float)
    # depth_img: np.ndarray of shape H x W
    # intrinsics: list of length 4; format: [fx, fy, cx, cy]
    rounded_pts = np.round(pts).astype(np.uint16)
    depths = depth_img[rounded_pts[:, 1], rounded_pts[:, 0]]
    depths = depths.astype(np.float32) / depth_scale_factor
    # valid_inds = depths > 0
    rounded_pts = rounded_pts[valid_inds]
    depths = depths[valid_inds]
    pts = pts[valid_inds].astype(np.float32)
    # ones = np.ones_like(pts[:, 1]).reshape(-1, 1)
    # pts_homo = np.concatenate([pts, ones], axis=-1)
    fx, fy, cx, cy = intrinsics
    _x = pts[:, 0]
    _y = pts[:, 1]
    Z = depths
    X = Z * (_x - cx) / fx
    Y = Z * (_y - cy) / fy
    backproj_pts = np.stack([X, Y, Z], axis=-1)
    return backproj_pts


def orthogonal_procrustes(src, tgt):
    # src: pointcloud (np.ndarray) N, 3
    # tgt: pointcloud (np.ndarray) N, 3
    # !! ASSUMES src and tgt ARE CORRESPONDING POINTS

    # Adapted from pytorch3d
    # https://github.com/facebookresearch/pytorch3d/blob/3388d3f0aa6bc44fe704fca78d11743a0fcac38c/pytorch3d/ops/points_alignment.py#L226

    # Compute mean of each pointset
    src_mu = src.mean(axis=0)
    tgt_mu = tgt.mean(axis=0)

    # Zero-center each pointset
    src_centered = src - src_mu
    tgt_centered = tgt - tgt_mu

    # Compute 3 x 3 covariance matrix
    src_tgt_cov = np.matmul(src_centered.T, tgt_centered)

    # SVD of cov mat gives us components of rotation
    U, D, V = np.linalg.svd(src_tgt_cov)

    # identity mat used for fixing reflections
    E = np.eye(3)
    # reflection test:
    #   checks whether the estimated rotation has det == 1
    #   if not, finds nearest rotation with det == 1 by
    #   flipping the sign of the last column of U
    R_test = np.matmul(U, V.T)
    det = np.linalg.det(R_test)
    if det == -1:
        print("REFLECTION DETECTED! FLIPPING SIGN OF LAST COL OF U!")
        E[-1, -1] = -1

    # rotation component
    R = np.matmul(np.matmul(U, E), V.T)
    # translation component
    t = tgt_mu - np.matmul(src_mu, R)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T


def procrustes(X, Y, scaling=False, reflection=False):
    """
    https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.0).sum()
    ssY = (Y0**2.0).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != "best":

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {"rotation": T, "scale": b, "translation": c}

    return d, Z, tform

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        default="capture-mar-28/calib",
        help="Path to log directory containing captured rgb-d images and extrinsics",
    )
   
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize each registered pair of pointclouds",
    )
    parser.add_argument(
        "--ref_cam", type=int, default=0, help="Index of reference camera"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    """
    OLD (Feb 17, 2024)
    top cam: 103422071983
    side cam (furthest from the computer): 233622075696
    side cam 2 (closest to the computer/TV): "102422072672"
    """

    """
    (20 March 2024)
    top cam: 103422071983
    left cam (furthest from computer): 027322071284
    right cam (closest to computer): 050522073498
    back cam (closest to the robot): 102422072672
    """

    serial_numbers = [
        "103422071983",
        "027322071284",
        "050522073498",
        # "102422072672"
    ]

    for serial_number in serial_numbers:
        capture_realsense_image(args.logdir, serial_number)

    # Seed RNG
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Index of reference camera (w.r.t. which all other cameras are calibrated)
    REF_CAM_IDX = args.ref_cam

    glob_str = os.path.join(args.logdir, "*")
    num_cameras = len(glob.glob(glob_str))
    # devids = sorted(glob.glob(glob_str))
    devids = [
        "103422071983",
        "027322071284",
        "050522073498",
        # "102422072672",
    ]
    print(f"Found {num_cameras} cameras...")

    tagsize = 0.03  # 3 cm tags
    num_tags = 213

    detector_options = apriltag.DetectorOptions(
        quad_decimate=1.0, refine_decode=True, refine_pose=True
    )
    detector = apriltag.Detector(options=detector_options)


    imgfiles = [
        os.path.join(args.logdir, devids[i], "rgb", str("0").zfill(5) + ".jpg")
        for i in range(num_cameras)
    ]
    depthfiles = [
        os.path.join(args.logdir, devids[i], "depth", str("0").zfill(5) + ".png")
        for i in range(num_cameras)
    ]
    depths = [cv2.imread(depthfile, cv2.IMREAD_ANYDEPTH) for depthfile in depthfiles]
    intrinsics = []
    for i in range(num_cameras):
        with open(os.path.join(args.logdir, devids[i], "camera_intrinsic.json")) as f:
            _intrinsic = json.load(f)
            intrinsics.append(
                {
                    "fx": _intrinsic["intrinsic_matrix"][0],
                    "fy": _intrinsic["intrinsic_matrix"][4],
                    "cx": _intrinsic["intrinsic_matrix"][6],
                    "cy": _intrinsic["intrinsic_matrix"][7],
                }
            )

    imgs = [cv2.imread(imgfile) for imgfile in imgfiles]
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    results = [detector.detect(gray) for gray in grays]

    camera_params = [
        [
            intrinsics[i]["fx"],
            intrinsics[i]["fy"],
            intrinsics[i]["cx"],
            intrinsics[i]["cy"],
        ]
        for i in range(num_cameras)
    ]
    
    viz_clouds = []  # This list will be set only if args.visualize is True

    # Poll all images for tags of interest
    img_tag_viewgraph = np.zeros((num_cameras, num_tags)).astype(bool)
    img_tag_confidences = np.zeros((num_cameras, num_tags)).astype(np.float32)
    img_tag_detection_map = np.zeros((num_cameras, num_tags)).astype(np.uint32)
    for cam_idx in range(num_cameras):
        for detection_idx, detection in enumerate(results[cam_idx]):
            tag_idx = detection.tag_id
            img_tag_viewgraph[cam_idx, tag_idx] = True
            img_tag_confidences[cam_idx, tag_idx] = detection.decision_margin
            img_tag_detection_map[cam_idx, tag_idx] = detection_idx

    # Compute the transform for each camera w.r.t. the reference camera
    tags_in_img_0 = img_tag_viewgraph[REF_CAM_IDX] == 1.0
    tags_in_img_0_inds = np.nonzero(tags_in_img_0)[
        0
    ]  # 0-th element cause np wraps additional dim
    tag_poses_cam_0 = np.zeros((num_tags, 4, 4))
    print(tags_in_img_0_inds)
    for tag_idx in tags_in_img_0_inds:
        detection_idx = img_tag_detection_map[REF_CAM_IDX, tag_idx]
        tag_poses_cam_0[tag_idx] = detection_to_metric_pose(
            detector,
            results[REF_CAM_IDX][detection_idx],
            camera_params[REF_CAM_IDX],
            tagsize,
        )
        # tag_poses_cam_0[tag_idx][:3, :3] = results_dt[0][detection_idx].pose_R
        # tag_poses_cam_0[tag_idx][:3, 3] = results_dt[0][detection_idx].pose_t[:3, 0] # pose_t is (3, 1)

    color_0 = o3d.geometry.Image(imgs[REF_CAM_IDX])
    depth_0 = o3d.geometry.Image(depths[REF_CAM_IDX])
    rgbd_0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_0, depth_0, convert_rgb_to_intensity=False
    )
    intrinsic_0 = o3d.camera.PinholeCameraIntrinsic(
        imgs[REF_CAM_IDX].shape[1],
        imgs[REF_CAM_IDX].shape[0],
        camera_params[REF_CAM_IDX][0],
        camera_params[REF_CAM_IDX][1],
        camera_params[REF_CAM_IDX][2],
        camera_params[REF_CAM_IDX][3],
    )
    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_0, intrinsic_0)
    pcd0_colors = np.asarray(pcd0.colors)
    pcd0_colors[:, 1:3] /= 2  # decrease opacity of G, B channels
    pcd0.colors = o3d.utility.Vector3dVector(pcd0_colors)
    
    if args.visualize:
        viz_clouds.append(pcd0)

    # Store pose of ref cam as the 4 x 4 identity matrix
    savefile = os.path.join(args.logdir, devids[REF_CAM_IDX], "pose.npy")
    np.save(savefile, np.eye(4))

    for cam_idx in range(num_cameras):
        if cam_idx == REF_CAM_IDX:
            continue
        print(f"Processing camera {cam_idx}...")
        tags_in_img_cur = img_tag_viewgraph[cam_idx] == 1.0
        common_tag_inds = np.nonzero(np.logical_and(tags_in_img_0, tags_in_img_cur))[
            0
        ]  

        cam_cur_to_cam_0_tf_measurements_rmat = []
        measurement_confidences = []
        _pts_0 = []
        _pts_cur = []
        for common_tag_idx in common_tag_inds:
            detection_idx_cur = img_tag_detection_map[cam_idx, common_tag_idx]
            tag_to_cam_cur_tf = detection_to_metric_pose(
                detector,
                results[cam_idx][detection_idx_cur],
                camera_params[cam_idx],
                tagsize,
            )
            detection_idx_0 = img_tag_detection_map[REF_CAM_IDX][common_tag_idx]
            _detection_0 = results[REF_CAM_IDX][detection_idx_0]
            _detection_cur = results[cam_idx][detection_idx_cur]

            _pts_0.append(np.asarray(_detection_0.center))
            _pts_cur.append(np.asarray(_detection_cur.center))
            cam_cur_to_cam_0_tf = np.matmul(
                tag_poses_cam_0[common_tag_idx], np.linalg.inv(tag_to_cam_cur_tf)
            )
            cam_cur_to_cam_0_tf_measurements_rmat.append(cam_cur_to_cam_0_tf)
            measurement_confidences.append(img_tag_confidences[cam_idx, detection_idx])

        _pts_0 = np.stack(_pts_0, axis=0)
        _pts_cur = np.stack(_pts_cur, axis=0)

        valid_inds = get_valid_points_in_both(
            _pts_0, depths[REF_CAM_IDX], _pts_cur, depths[cam_idx]
        )

        _corner_pcd_0 = backproject_to_3d(
            _pts_0, depths[REF_CAM_IDX], camera_params[REF_CAM_IDX], valid_inds
        )
        _corner_pcd_cur = backproject_to_3d(
            _pts_cur, depths[cam_idx], camera_params[cam_idx], valid_inds
        )

        # this uses the reverse convention
        _, _, tform = procrustes(_corner_pcd_cur, _corner_pcd_0)
        _transform = np.eye(4)
        _transform[:3, :3] = tform["rotation"].T
        _src_rotated = np.matmul(_corner_pcd_cur, tform["rotation"].T)
        _translation = _corner_pcd_0.mean(0) - _src_rotated.mean(0)
        _transform[:3, 3] = _translation
        _corner_pcd_cur = _src_rotated + _translation

        color_cur = o3d.geometry.Image(imgs[cam_idx])
        depth_cur = o3d.geometry.Image(depths[cam_idx])
        rgbd_cur = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_cur, depth_cur, convert_rgb_to_intensity=False
        )
        intrinsic_cur = o3d.camera.PinholeCameraIntrinsic(
            imgs[cam_idx].shape[1],
            imgs[cam_idx].shape[0],
            camera_params[cam_idx][0],
            camera_params[cam_idx][1],
            camera_params[cam_idx][2],
            camera_params[cam_idx][3],
        )
        pcd_cur = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_cur, intrinsic_cur
        )

        pcd0_corners = o3d.geometry.PointCloud()
        pcd0_corners.points = o3d.utility.Vector3dVector(_corner_pcd_0)
        pcd0_corners.paint_uniform_color([1, 0, 0])

        pcd_cur_corners = o3d.geometry.PointCloud()
        pcd_cur_corners.points = o3d.utility.Vector3dVector(_corner_pcd_cur)
        pcd_cur_corners.paint_uniform_color([0, 0, 1])
        
        # This seems to work well
        _rot = _transform[:3, :3]
        _translation = _transform[:3, 3]
        _transform[:3, :3] = _rot.T
        _transform[:3, 3] = _translation
        pcd_cur.transform(_transform)
        
        pcd0_colors[:, :2] /= 2  # decrease opacity of R, G channels
        pcd0.colors = o3d.utility.Vector3dVector(pcd0_colors)

        if args.visualize:
            o3d.visualization.draw_geometries([pcd0, pcd_cur])
            viz_clouds.append(pcd_cur)

        savefile = os.path.join(args.logdir, devids[cam_idx], "pose.npy")
        np.save(savefile, _transform)
        print(f"Saved transform to {savefile}")
    
    if args.visualize:
        o3d.visualization.draw_geometries(viz_clouds)

if __name__ == "__main__":
    main()
