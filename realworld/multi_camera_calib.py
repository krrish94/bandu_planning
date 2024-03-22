import copy
import json
import os
import pickle as pkl

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from gen_stack_exp import merge_pcds
from PIL import Image
from tqdm import tqdm, trange


def capture_realsense_image(output_folder, serial_number):

    from types import SimpleNamespace

    from realsense_recorder import Preset, get_profiles, realsense_capture

    rs_args = SimpleNamespace()
    rs_args.output_folder = output_folder
    rs_args.realsense_preset = 1
    rs_args.clipping_distance = 3
    rs_args.frames_to_capture = 1
    rs_args.render_images = False
    rs_args.record_images = True
    rs_args.color_profile = 14  # 42
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

    print(f"Using the profiles: color: {color_profile}, depth: {depth_profile}")
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

    realsense_capture(
        rs_args, pipeline, depth_sensor, align, serial_number, frames_to_skip=10
    )


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
        with open(intrinsic_json, "r") as json_file:
            intrinsic_data = json.load(json_file)
        width = intrinsic_data["width"]
        height = intrinsic_data["height"]
        intrinsic_matrix = intrinsic_data["intrinsic_matrix"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic_matrix[0],
            intrinsic_matrix[4],
            intrinsic_matrix[6],
            intrinsic_matrix[7],
        )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img,
            o3d.geometry.Image(depth_img_np),
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        # o3d.visualization.draw_geometries([pcd])
        pcds.append(pcd)
    return pcds


def register_icp(pcds, init=None, debug_visualization=False):
    camera_order = [0, 1]  # TODO: read the camera order from the config file
    # TODO: Error handling if number of poses in the camera order < 1
    # camera_order = reconstruction_config['realsense_camera_order'] # defines the order in which the poses are processed by ICP --> for good initialization

    registered_pcd = pcds[camera_order[0]]
    registered_pcd.estimate_normals()
    transformations = dict()  # transformation matrices representing poses
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
            current_pcd,
            registered_pcd,
            max_correspondence_distance=0.05,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        # transform the current_pcd by the estimated transformation
        current_pcd = current_pcd.transform(reg_p2p.transformation)
        transformations[camera_pose] = np.asarray(
            reg_p2p.transformation, dtype=np.float32
        )
        # merge the pcds to get the registered pcd for the current step
        registered_pcd = merge_pcds([current_pcd, registered_pcd])
        registered_pcd.estimate_normals()
        if (
            debug_visualization
        ):  # visualize the registered pcd at the current step if debug_visualization is set to True
            o3d.visualization.draw_geometries([registered_pcd])

    return transformations, registered_pcd


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def execute_fast_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    # % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])  # ,
    #   zoom=0.4559,
    #   front=[0.6452, -0.3036, -0.7011],
    #   lookat=[1.9892, 2.0208, 1.8945],
    #   up=[-0.2779, -0.9482, 0.1556])


def global_registration(source, target, voxel_size=0.05, visualize=False, fast=False):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    if fast:
        result_ransac = execute_fast_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )
    else:
        result_ransac = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )
    # print(result_ransac)
    if visualize:
        draw_registration_result(source_down, target_down, result_ransac.transformation)
    return result_ransac


def run_search_for_calibration(source, target, savedir, iters):

    fitnesses = []
    registration_results = []
    inlier_rmses = []

    for i in trange(iters):
        registration_result = global_registration(
            source, target, voxel_size=0.01, visualize=False
        )
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


def pick_points_from_gui(pcd: o3d.geometry.PointCloud) -> np.array:
    """Return a set of clicked points from the visualizer.

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


def main():
    imgdir = "capture-feb-17/dry-erase-marker-2"
    """
    top cam: 103422071983
    side cam (furthest from the computer): 233622075696
    side cam 2 (closest to the computer/TV): "102422072672"
    """

    # "103422071983", "233622075696",
    serial_numbers = [
        "103422071983",
        "233622075696",
        "102422072672",
    ]

    for serial_number in serial_numbers:
        capture_realsense_image(imgdir, serial_number)

    # tuned depth cutoffs;
    # cam1 -> 1.5, cam2 -> 1.3, cam3 -> 0.65
    depth_cutoffs = [1.5, 1.3, 0.65]
    pcds = read_pcds(imgdir, serial_numbers, depth_cutoffs)
    # transformations, registered_pcd = register_icp(pcds, debug_visualization=True)
    # picked_points = pick_points_from_gui(pcds[1])
    # quit()

    pcd1 = pcds[0]
    pcd2 = pcds[1]
    pcd3 = pcds[2]

    cam1_to_cam2 = np.array(
        [
            [-0.12, -0.99, -0.0, -0.12],
            [0.57, -0.07, 0.82, -0.82],
            [-0.81, 0.1, 0.58, -0.01],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cam1_to_cam3 = np.array(
        [
            [0.44, 0.9, 0.01, 0.04],
            [-0.59, 0.28, 0.77, -0.76],
            [0.69, -0.34, 0.64, -0.16],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pcd1.estimate_normals()
    pcd2.estimate_normals()
    pcd3.estimate_normals()
    pcd2.transform(np.linalg.inv(cam1_to_cam2))
    pcd3.transform(np.linalg.inv(cam1_to_cam3))

    # Refine further using ICP
    result_icp_cam_2_to_1 = o3d.pipelines.registration.registration_icp(
        pcd2,
        pcd1,
        0.03,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    result_icp_cam_3_to_1 = o3d.pipelines.registration.registration_icp(
        pcd3,
        pcd1,
        0.03,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    pcd2.transform(result_icp_cam_2_to_1.transformation)
    pcd3.transform(result_icp_cam_3_to_1.transformation)
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

    # for i in range(10):
    #     registration_result = global_registration(source, target, voxel_size=0.01, visualize=True)
    #     print(registration_result.transformation)
    """
    Run global registration multiple times and save valid results to pkl file
    """
    # run_search_for_calibration(source, target, imgdir, iters=500)
    """
    Pick points from GUI
    """
    # picked_points = pick_points_from_gui(pcds[0])
    # print(picked_points)
    """
    Process global reg results and visualize most promising solutions
    """
    # global_reg_results = None
    # with open(os.path.join(imgdir, "global_reg_results.pkl"), "rb") as f:
    #     global_reg_results = pkl.load(f)
    # print(global_reg_results.keys())
    # print(np.asarray(global_reg_results["fitnesses"]).argmax(), np.asarray(global_reg_results["fitnesses"]).max())
    # print(np.asarray(global_reg_results["inlier_rmses"]).argmin(), np.asarray(global_reg_results["inlier_rmses"]).min())
    # print(global_reg_results["fitnesses"])

    # num_results_to_viz = 25
    # sampled_inds = np.random.choice(np.arange(len(global_reg_results["fitnesses"])), size=num_results_to_viz, replace=False)
    # print(sampled_inds)

    # for idx in sampled_inds:
    #     print(global_reg_results["registration_results"][idx])
    #     print(global_reg_results["fitnesses"][idx])
    #     draw_registration_result(source, target, global_reg_results["registration_results"][idx])

    # max_fitness_idx = np.asarray(global_reg_results["fitnesses"]).argmax()
    # min_inlier_rmse_idx = np.asarray(global_reg_results["inlier_rmses"]).argmin()
    # draw_registration_result(source, target, global_reg_results["registration_results"][max_fitness_idx])
    """
    Refine (local reg) using ICP
    """
    # # # cam1 -> cam2
    # # init_guess = np.array([
    # #     [-0.12, -0.99,   -0.0, -0.12],
    # #     [ 0.57, -0.07,  0.82, -0.82],
    # #     [-0.81,  0.1 , 0.58, -0.01],
    # #     [ 0.,    0.,    0.,    1.  ]
    # # ])
    # # cam1 -> cam3
    # init_guess = np.array([
    #     [ 0.53,  0.85, -0.07,  0.11],
    #     [-0.62,  0.44,  0.66, -0.64],
    #     [ 0.59, -0.3,   0.75, -0.27],
    #     [ 0.,    0.,    0.,    1.  ],
    # ])
    # # transformations, registered_pcd = register_icp(pcds, init=init_guess, debug_visualization=True)
    # source.estimate_normals()
    # target.estimate_normals()
    # result_icp = o3d.pipelines.registration.registration_icp(source, target, 0.03, init_guess, o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(result_icp.transformation)
    # draw_registration_result(source, target, result_icp.transformation)


if __name__ == "__main__":
    main()
