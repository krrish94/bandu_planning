""" High-level manipulation wrapper for franka-robot
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2
import time
from typing import Tuple
from bandu_stacking.robot.cri.cri.robot import SyncRobot
from bandu_stacking.robot.cri.cri.controller import PyfrankaController
import cv2
import numpy as np
import json
import os
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial.transform import Rotation as R
import shutil
import math

## ==== DLO Utills ==== ##


HSV_PREDEFINE = {
    "blue": {"low": np.array([100, 196, 120]), "high": np.array([125, 255, 255])},
    "black": {"low": np.array([0, 0, 0]), "high": np.array([255, 255, 80])},
    "orange": {"low": np.array([0, 100, 100]), "high": np.array([40, 255, 255])},
}


def color_segment(image, low_hsv, high_hsv, enable_vis=False):
    """Segmentation using color range
    Practice from https://realpython.com/python-opencv-color-spaces/
    Notice: the image should be in RGB
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # convert to HSV space

    # Conduct segment
    mask = cv2.inRange(hsv_image, low_hsv, high_hsv)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    if enable_vis:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(masked_image)
        plt.show()
    return mask, masked_image


## ==== Image Utils ==== ##


def draw_bbox_interactively(image):
    # Global variables to store coordinates
    drawing = False
    bbox_start = (-1, -1)
    bbox_end = (-1, -1)

    vis_image = image.copy()

    def draw_bbox(event, x, y, flags, param):
        nonlocal drawing, bbox_start, bbox_end

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            bbox_start = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            bbox_end = (x, y)

            # Draw the bounding box on the image
            cv2.rectangle(vis_image, bbox_start, bbox_end, (0, 255, 0), 2)
            cv2.imshow("Interactive Bounding Box", vis_image)
            cv2.waitKey(0)

    cv2.namedWindow("Interactive Bounding Box")
    cv2.setMouseCallback("Interactive Bounding Box", draw_bbox)

    # Main loop
    while True:
        cv2.imshow("Interactive Bounding Box", vis_image)

        # Press 'Esc' to exit the interactive process
        if cv2.waitKey(1) == 27:
            break

    # After the loop, destroy all windows
    cv2.destroyAllWindows()

    # return (x0, y0, width, height)
    bbox_width = bbox_end[0] - bbox_start[0]
    bbox_height = bbox_end[1] - bbox_start[1]
    return [bbox_start[0], bbox_start[1], bbox_width, bbox_height]


def reisze_and_normalize_image(image, image_size):
    """Resize the image to the specified size while maintaining the aspect ratio."""
    # Get the current dimensions of the input image
    height, width = image.shape[:2]

    # Calculate the scaling factors for width and height
    scale_x = image_size / width
    scale_y = image_size / height

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, (int(width * scale_x), int(height * scale_y)))

    # Normalize to (0, 1)
    resized_image = (resized_image.astype(np.float32)) / 255.0
    return resized_image


## ==== BOP Utils ==== ##


def save_to_bop(
    bop_path,
    data_split,
    scene_id,
    obj_ids,
    color_list,
    depth_list,
    time_stamps,
    intrinsic,
    draw_bbox=False,
):
    """Save images and depth into BOP
    Args:
        intrinsic: a 3x3 matrix
    """
    bop_split_path = os.path.join(bop_path, data_split)
    scene_path = os.path.join(bop_split_path, f"{scene_id:06}")
    scene_rgb_path = os.path.join(scene_path, "rgb")
    scene_rgb_roi_path = os.path.join(scene_path, "rgb_roi")
    scene_depth_path = os.path.join(scene_path, "depth")
    scene_mask_path = os.path.join(scene_path, "mask")
    scene_mask_visb_path = os.path.join(scene_path, "mask_visib")
    if os.path.exists(scene_path):
        shutil.rmtree(scene_path)
    os.makedirs(scene_path, exist_ok=True)
    os.makedirs(scene_rgb_path, exist_ok=True)
    os.makedirs(scene_rgb_roi_path, exist_ok=True)
    os.makedirs(scene_depth_path, exist_ok=True)
    os.makedirs(scene_mask_path, exist_ok=True)
    os.makedirs(scene_mask_visb_path, exist_ok=True)

    num_images = len(color_list)
    # save camera info
    camera_data = {}
    for i in range(num_images):
        camera_data[i] = {}
        camera_data[i]["depth_scale"] = 1.0
        camera_data[i]["cam_K"] = intrinsic.tolist()
    with open(os.path.join(scene_path, "scene_camera.json"), "w") as f:
        json.dump(camera_data, f)

    # fake gt & gt_info file
    height, width, _ = color_list[0].shape
    scene_gt = {}
    scene_gt_info = {}
    bbox_list = []
    for i in range(num_images):
        scene_gt[i] = []
        scene_gt_info[i] = []
        for j, obj_id in enumerate(obj_ids):  # object_id is 1-based
            # pose
            scene_gt[i].append(
                {
                    "cam_R_m2c": np.eye(3).tolist(),
                    "cam_t_m2c": np.zeros(3).tolist(),
                    "obj_id": obj_id,
                }
            )
            # bbox
            if i == 0:
                # Set bbox in the first image
                if draw_bbox:
                    bbox = draw_bbox_interactively(color_list[0])
                else:
                    bbox = [0, 0, width, height]
                bbox_list.append(bbox)
            scene_gt_info[i].append(
                {
                    "bbox_obj": bbox_list[j],
                    "bbox_visib": bbox_list[j],
                    "px_count_all": height * width,
                    "px_count_valid": height * width,
                    "visib_fract": 1.0,
                }
            )
    with open(os.path.join(scene_path, "scene_gt.json"), "w") as f:
        json.dump(scene_gt, f)
    with open(os.path.join(scene_path, "scene_gt_info.json"), "w") as f:
        json.dump(scene_gt_info, f)

    # save rgb & depth image
    for i, (color_image, depth_image) in enumerate(zip(color_list, depth_list)):
        cv2.imwrite(os.path.join(scene_rgb_path, f"{i:06}.jpg"), color_image)
        cv2.imwrite(os.path.join(scene_depth_path, f"{i:06}.png"), depth_image)
        print(f"Saving {os.path.join(scene_rgb_path, f'{i:06}.jpg')}...")
        print(f"Saving {os.path.join(scene_depth_path, f'{i:06}.jpg')}...")
        # save rgb roi image
        if bbox is not None:
            x0, y0, w, h = bbox
            cropped_image = color_image[y0 : y0 + h, x0 : x0 + w]
            cv2.imwrite(os.path.join(scene_rgb_roi_path, f"{i:06}.png"), cropped_image)

    full_mask = np.ones([height, width], dtype=np.uint8) * 255
    # save fake mask & mask_visb
    for i in range(num_images):
        for j in range(len(obj_ids)):
            print(f"Saving {os.path.join(scene_mask_path, f'{i:06}_{j:06}.png')}...")
            print(
                f"Saving {os.path.join(scene_mask_visb_path, f'{i:06}_{j:06}.png')}..."
            )
            cv2.imwrite(os.path.join(scene_mask_path, f"{i:06}_{j:06}.png"), full_mask)
            cv2.imwrite(
                os.path.join(scene_mask_visb_path, f"{i:06}_{j:06}.png"), full_mask
            )

    # save time stamps
    with open(os.path.join(scene_path, "time_stamps.json"), "w") as f:
        json.dump(time_stamps, f)


## ==== Transform Utils ==== ##


def euler2mat(pose_e, rot_unit="deg", trans_unit="mm") -> np.ndarray:
    """Convert euler pose to matrix"""
    x, y, z, ex, ey, ez = pose_e
    if rot_unit == "deg":
        ex = math.radians(ex)
        ey = math.radians(ey)
        ez = math.radians(ez)
    elif rot_unit == "rad":
        ex = ex
        ey = ey
        ez = ez

    if trans_unit == "mm":
        x = x / 1000.0
        y = y / 1000.0
        z = z / 1000.0
    elif trans_unit == "m":
        x = x
        y = y
        z = z

    Rx = np.array(
        [[1, 0, 0], [0, math.cos(ex), -math.sin(ex)], [0, math.sin(ex), math.cos(ex)]]
    )
    Ry = np.array(
        [[math.cos(ey), 0, math.sin(ey)], [0, 1, 0], [-math.sin(ey), 0, math.cos(ey)]]
    )
    Rz = np.array(
        [[math.cos(ez), -math.sin(ez), 0], [math.sin(ez), math.cos(ez), 0], [0, 0, 1]]
    )
    R = Rz @ Ry @ Rx
    # R = np.eye(3, dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, z])

    return T


def mat2euler(pose_m):
    """Convert matrix to euler pose"""
    R = pose_m[:3, :3]
    trans = pose_m[:3, 3]
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    rot_euler = np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])
    trans_euler = trans * 1000.0  # Convert to m
    pose_euler = np.hstack([trans_euler, rot_euler])
    return pose_euler.astype(np.float64)


def quat2mat(pos_quat, rot_unit="deg", trans_unit="mm"):
    """Convert the (pos, quat) representation to mat"""
    T = np.eye(4, dtype=np.float32)
    if trans_unit == "mm":
        T[:3, 3] = pos_quat[:3] / 1000.0
    elif trans_unit == "m":
        T[:3, 3] = pos_quat[:3]
    else:
        raise ValueError(f"trans_unit: {trans_unit} is not defined!")

    T[:3, :3] = R.from_quat(pos_quat[3:]).as_matrix()
    return T.astype(np.float32)


def compute_angular_velocity(R_s, R_g, t):
    """Rotate from R_s to R_t using time t, return angular velocity"""
    R_gs = R_g @ np.linalg.inv(R_s)
    axis, angle = rotation_matrix_to_axis_angle(R_gs)
    if np.abs(angle) < 1e-6:
        return np.zeros(
            3,
        )
    angular_vel = axis * angle / t
    return angular_vel


def compute_angular_period(R_s, R_g, vel_a):
    """Compute the minimum period from R_s to R_g under vel_a"""
    R_gs = R_g @ np.linalg.inv(R_s)
    axis, angle = rotation_matrix_to_axis_angle(R_gs)
    return angle / vel_a


def rotation_matrix_to_axis_angle(R):
    """Convert a rotation matrix to axis-angle representation"""
    angle = np.arccos((np.trace(R) - 1) / 2)
    x = (R[2, 1] - R[1, 2]) / (2 * np.sin(angle) + 1e-6)
    y = (R[0, 2] - R[2, 0]) / (2 * np.sin(angle) + 1e-6)
    z = (R[1, 0] - R[0, 1]) / (2 * np.sin(angle) + 1e-6)
    return np.array([x, y, z]), angle


## ==== File Utils ==== ##


def read_robot_state_from_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        json_objects = [json.loads(line) for line in lines]
    return json_objects


## ==== Vis Utils === ##


def o3d_create_plane():
    """Create an o3d plane"""
    # Define vertices of the plane
    vertices = np.array(
        [
            [-0.5, -0.5, 0.0],  # Vertex 0
            [0.5, -0.5, 0.0],  # Vertex 1
            [-0.5, 0.5, 0.0],  # Vertex 2
            [0.5, 0.5, 0.0],  # Vertex 3
        ]
    )
    # Define triangles for both sides of the plane
    triangles_front = np.array(
        [
            [0, 1, 2],  # Triangle 0 (vertices 0, 1, 2)
            [1, 3, 2],  # Triangle 1 (vertices 1, 3, 2)
        ]
    )
    triangles_back = np.array(
        [
            [1, 0, 2],  # Triangle 0 (vertices 1, 0, 2)
            [3, 1, 2],  # Triangle 1 (vertices 3, 1, 2)
        ]
    )

    # Create TriangleMesh objects for both sides of the plane
    mesh_front = o3d.geometry.TriangleMesh()
    mesh_back = o3d.geometry.TriangleMesh()

    # Set vertices and triangles for both sides
    mesh_front.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_back.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_front.triangles = o3d.utility.Vector3iVector(triangles_front)
    mesh_back.triangles = o3d.utility.Vector3iVector(triangles_back)

    # Combine the two meshes into a single geometry object
    double_sided_plane = mesh_front
    double_sided_plane += mesh_back
    return double_sided_plane


def create_cube(size=1.0):
    """Create a cube with the given size"""
    # The center of the cube will be the origin
    half_size = size / 2.0
    vertices = [
        [-half_size, -half_size, -half_size],
        [-half_size, -half_size, half_size],
        [-half_size, half_size, -half_size],
        [-half_size, half_size, half_size],
        [half_size, -half_size, -half_size],
        [half_size, -half_size, half_size],
        [half_size, half_size, -half_size],
        [half_size, half_size, half_size],
    ]
    triangles = [
        [0, 1, 2],  # Instead of [0, 2, 1]
        [1, 3, 2],  # Instead of [1, 2, 3]
        [4, 6, 5],  # Instead of [4, 5, 6]
        [5, 6, 7],  # Instead of [5, 7, 6]
        [0, 4, 1],  # Instead of [0, 1, 4]
        [1, 4, 5],  # Instead of [1, 5, 4]
        [2, 3, 6],  # Instead of [2, 6, 3]
        [3, 7, 6],  # Instead of [3, 6, 7]
        [0, 2, 4],  # Instead of [0, 4, 2]
        [2, 6, 4],  # Instead of [2, 4, 6]
        [1, 5, 3],  # Instead of [1, 3, 5]
        [3, 5, 7],  # Instead of [3, 7, 5]
    ]
    # Colors for the 6 faces
    face_colors = [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
    ]
    # Apply the color to each triangle (each face has two triangles)
    vertex_colors = []
    for i in range(0, len(triangles), 2):
        face_color = face_colors[i // 2]
        for _ in range(6):  # 3 vertices * 2 triangles
            vertex_colors.append(face_color)

    cube = o3d.geometry.TriangleMesh()
    cube.vertices = o3d.utility.Vector3dVector(vertices)
    cube.triangles = o3d.utility.Vector3iVector(triangles)
    cube.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    cube.compute_vertex_normals()

    return cube


def pointcloud_to_dict(pc: o3d.geometry.PointCloud) -> dict:
    return {
        "points": np.asarray(pc.points),
        "colors": np.asarray(pc.colors),
        # Add other attributes of PointCloud if needed, like normals, etc.
    }


def dict_to_pointcloud(data: dict) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data["points"])
    pc.colors = o3d.utility.Vector3dVector(data["colors"])
    # Retrieve other attributes if saved, like normals, etc.
    return pc


def show_traj(
    pose_list: list, color_list: list | None = None, pose_type: str = "euler", size=0.1
):
    """Show trajectories in open3d, euler is the default format"""
    vis_list = []
    for i, pose in enumerate(pose_list):
        if pose_type == "euler":
            assert pose.shape == (6,) or pose.shape == (6, 1)
            pose_mat = euler2mat(pose)
        else:
            assert pose.shape == (4, 4)
            pose_mat = pose
        pose_geo = o3d.geometry.TriangleMesh.create_coordinate_frame(size)
        if color_list is not None:
            pose_geo.paint_uniform_color(color_list[i])
        pose_geo.transform(pose_mat)
        vis_list.append(pose_geo)
    o3d.visualization.draw_geometries(vis_list)


## ==== Geometry Utils ==== ##


def load_socket_connector(data_folder: str, model_type: str):
    """load geometry model, only for cable_manip"""
    cad_dict = {
        "amp_socket": 1,
        "amp_connector": 2,
        "atx_socket": 3,
        "atx_connector": 4,
    }
    model_folder = os.path.join(data_folder, "models")
    socket_name = f"{model_type}_socket"
    connector_name = f"{model_type}_connector"

    socket_mesh = o3d.io.read_triangle_mesh(
        os.path.join(model_folder, f"obj_{cad_dict[socket_name]:06}.ply")
    )
    socket_mesh.compute_triangle_normals()
    socket_mesh.paint_uniform_color([0.5, 0.0, 0.0])
    connector_mesh = o3d.io.read_triangle_mesh(
        os.path.join(model_folder, f"obj_{cad_dict[connector_name]:06}.ply")
    )
    connector_mesh.compute_triangle_normals()
    connector_mesh.paint_uniform_color([0.0, 0.5, 0.0])
    # scale
    socket_mesh.scale(0.001, center=(0, 0, 0))  # convert mm to m
    connector_mesh.scale(0.001, center=(0, 0, 0))  # convert mm to m

    return socket_mesh, connector_mesh


def generate_video(frames, video_name="video.mp4"):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"mp4v"), 1, (width, height)
    )

    # Loop to add images to video
    for frame in frames:
        video.write(frame)

    # Release the video writer
    video.release()
    cv2.destroyAllWindows()

    return video


## ==== Action Utils ==== ##


def interpolate_action(cur_pose: np.ndarray, goal_pose: np.ndarray, num_step: int):
    """cur_pose, goal_pose are all transition"""
    diff_pose = goal_pose - cur_pose  # (action_dim)
    step_size = 1.0 / num_step
    steps = np.linspace(start=0.5, stop=1.0, num=num_step)
    actions = cur_pose[None, :] + diff_pose[None, :] * steps[:, None]  # (num_step, action_dim)
    return actions

## ==== Print Utills ==== ##


def print_pose(pose: np.ndarray):
    pose_str =  [str(v) for v in pose.flatten().tolist()]
    print(",".join(pose_str))


def get_pointcloud(depth, intrinsic):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsic[0, 2]) * (depth / intrinsic[0, 0])
    py = (py - intrinsic[1, 2]) * (depth / intrinsic[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


## ==== Noise Utils === ##

import numpy as np
from scipy.spatial.transform import Rotation as R

def apply_random_noise_to_transform(T, r_noise_level, t_noise_level, seed):
    """Apply a random noise to SE3 transformation with a given seed.

    Args:
        T: 4x4 transformation matrix representing SE3 transformation.
        r_noise_level: Level of random noise to apply to rotation (in radians).
        t_noise_level: Level of random noise to apply to translation (in meters).
        seed: Seed for random number generation.

    Returns:
        T_noisy: Noisy SE3 transformation matrix.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Extract rotation and translation components from the SE3 transformation
    R_matrix = T[:3, :3]  # Rotation matrix
    translation = T[:3, 3]  # Translation vector

    # Generate random noise for rotation (using axis-angle representation)
    r_noise_axis_angle = np.random.uniform(-1, 1, 3)  # Random axis
    r_noise_axis_angle /= np.linalg.norm(r_noise_axis_angle)  # Normalize axis
    r_noise_angle = np.random.uniform(-r_noise_level, r_noise_level)  # Random angle
    r_noise_rotation = R.from_rotvec(r_noise_angle * r_noise_axis_angle)

    # Apply random noise to the rotation component
    R_noisy_matrix = r_noise_rotation.as_matrix() @ R_matrix

    # Generate random noise for translation
    t_noise = np.random.uniform(-t_noise_level, t_noise_level, 3)

    # Apply random noise to the translation component
    translation_noisy = translation + t_noise

    # Create the noisy SE3 transformation matrix
    T_noisy = np.eye(4)
    T_noisy[:3, :3] = R_noisy_matrix
    T_noisy[:3, 3] = translation_noisy

    return T_noisy



HOME_JOINT = (0.01, -44.87, 0.03, -135.02, -0.1, 89.96, 44.98)
TOP_DOWN_R = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, -1.0, -0.0],
     [0.0, 0.0, -1.0]])
HOVER_HEIGHT = 0.1
INSERT_DEPTH = 0.1
MAX_TRY = 10
GRASP_WIDTH = 0.008  # m
EPS_INNER = 0.1
EPS_OUTER = 0.1
RELEASE_WIDTH = 0.1
GRASP_SPEED = 0.05  # m/s
GRASP_FORCE = 50.0  # N
MIN_TIME_STEP = 0.005  # sec (200HZ)
LINER_SPEED_LIMIT = 0.1  # m/s
ANGULAR_SPEED_LIMIT = 0.5  # radius/s
JOINT_SPEED_LIMIT = 1.0


class FrankaFr3:
    """Robot Manipulator, using CRI as control interface"""

    def __init__(self, robot_ip: str, c_T_x: np.ndarray=np.eye(4, dtype=np.float32), calib_type: str = "c2f"):
        """Different calibration type will influence your final results.
            c2f: camera to flange. We use this mode, when camera is mounted on the wrist.
        """
        self.robot = SyncRobot(PyfrankaController(ip=robot_ip))
        self.controller = self.robot.controller
        self.gripper = self.controller.gripper

        # Set parameters
        self.robot.axes = 'sxyz'
        # self.robot.tcp = (0, 0, 0, 0, 0, 0)

        self.controller.set_joint_impedance(
            (3000, 3000, 3000, 2500, 2500, 2000, 2000))
        self.controller.rel_velocity = 0.1
        self.controller.rel_accel = 0.1
        self.controller.rel_jerk = 0.1

        # Grasp-related: Hover matrix
        self.is_grasping = False
        self.h_T_g = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, HOVER_HEIGHT],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.g_T_h = np.linalg.inv(self.h_T_g)

        # Insert-related: Insert matrix
        r_o_T_e = R.from_euler('XYZ', np.array(
            [0.0, np.pi/2.0, np.pi])).as_matrix()  # object to ee (default)
        self.o_T_e = np.eye(4, dtype=np.float64)
        self.o_T_e[:3, :3] = r_o_T_e

        self.i_T_g = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -INSERT_DEPTH],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.g_T_i = np.linalg.inv(self.i_T_g)
        self.update_calib(c_T_x, calib_type)

        # Internal variable
        self.prev_linear_vel = np.zeros((3,), dtype=np.float32)  # (vx, vy, vz)
        self.prev_rot_vel = np.zeros((3,), dtype=np.float32)  # (wx, wy, wz)

        print(euler2mat(self.robot.pose))
        print("Franka is initialized...")

    def reset(self, joint_val: tuple | None = None, pose: np.ndarray | None = None, with_gripper: bool = True):
        if pose is None:
            joint_val_d = joint_val if joint_val is not None else HOME_JOINT
            self.move_joints(joint_val_d)
            if with_gripper:
                self.gripper.homing()
        else:
            assert pose.shape == (
                4, 4), "Pose should be a transformation matrix"
            self.move_linear(pose)
        self.update_tf()
        print("Franka is reset to home position...")

    #### Calibration related ####
    def update_calib(self, c_T_x, calib_type):
        self.calib_type = calib_type
        if calib_type == "c2b":  # camera to base
            self.c_T_b = c_T_x
            self.b_T_c = np.linalg.inv(self.c_T_b)  # base to camera
            self.c_T_f = None
            self.f_T_c = None
        elif calib_type == "c2f":  # camera to flange
            self.c_T_f = c_T_x
            self.f_T_c = np.linalg.inv(self.c_T_f)
            self.c_T_b = None
            self.b_T_c = None
        elif calib_type == "c2ee":  # camera to ee
            self.c_T_ee = c_T_x
            self.ee_T_c = np.linalg.inv(self.c_T_ee)
            self.c_T_b = None
            self.b_T_c = None
        else:
            raise ValueError(f"{calib_type} is not supported yet!")
        self.update_tf()

    def update_tf(self):
        if self.calib_type == "c2f":
            f_T_ee = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.1],
                               [0.0, 0.0, 0.0, 1.0]])
            b_T_ee = euler2mat(self.pose)
            b_T_f = b_T_ee @ np.linalg.inv(f_T_ee)
            # b_T_f = euler2mat(self.pose)
            self.b_T_c = b_T_f @ self.f_T_c
            self.c_T_b = np.linalg.inv(self.b_T_c)
        elif self.calib_type == "c2ee":
            b_T_ee = euler2mat(self.pose)
            self.b_T_c = b_T_ee @ self.ee_T_c
            self.c_T_b = np.linalg.inv(self.b_T_c)

    #### Task-related action ####
    def grasp_at(self, pos: np.ndarray, periods: list, coords: str = 'robot'):
        """Args:
            - coords: coordinate should be 'robot' or 'camera'
        """
        if coords == 'camera':
            self.update_tf()
            if pos.shape == (4, 4):
                b_T_g = self.b_T_c @ pos
            elif pos.shape == (3,) or pos.shape == (3, 1):
                pos = np.concatenate([pos.reshape([3, 1]), np.ones([1, 1])])
                pos = self.b_T_c @ pos
                b_T_g = np.eye(4, dtype=np.float64)
                b_T_g[:3, :3] = TOP_DOWN_R  # By default use top-down
                b_T_g[:3, 3] = pos[:3, 0]
            else:
                raise ValueError("pos shape is not supported.")
        elif coords == 'robot':
            if pos.shape == (3,) or pos.shape == (3, 1):  # Only position is provided
                b_T_g = np.eye(4, dtype=np.float64)
                b_T_g[:3, :3] = TOP_DOWN_R  # By default use top-down
                b_T_g[:3, 3] = pos
            elif pos.shape == (6,) or pos.shape == (6, 1):
                b_T_g = euler2mat(pos)
            elif pos.shape == (4, 4):
                b_T_g = pos
            else:
                raise ValueError("pos shape is not supported.")
        else:
            raise ValueError("Coords has to be camera or robot.")
        b_T_h = b_T_g @ self.g_T_h

        trajs = []
        assert len(periods) == 3  # period is provided for each step
        # move to the hold position
        trajs.append(self.move_linear_within_time(b_T_h, periods[0]))

        # move to the grasp position
        trajs.append(self.move_linear_within_time(b_T_g, periods[1]))

        # grasp
        self.gripper.grasp(GRASP_WIDTH, GRASP_SPEED,
                           GRASP_FORCE, EPS_INNER, EPS_OUTER)

        # move back to hold position
        trajs.append(self.move_linear_within_time(b_T_h, periods[2]))
        return trajs

    def insert_at(self, pos: np.ndarray, periods, coords: str = 'robot'):
        """Args:
            - pose: a coordinate frame located at the center of the hole, 
                with z-axis pointing out, y-axis along the gripper's y-axis
            - coords: coordinate should be 'robot' or 'camera'
        """
        if pos.shape == (6,) or pos.shape == (6, 1):
            pos = euler2mat(pos)
        assert pos.shape == (4, 4), "Insertion pos must be a (6,) or (4, 4)"

        if coords == 'camera':
            b_T_g = self.b_T_c @ pos
        else:
            b_T_g = pos
        b_T_i = b_T_g @ self.g_T_i  # pre-insert object pose

        b_T_ge = b_T_g @ self.o_T_e  # ee pose of b_T_g
        b_T_ie = b_T_i @ self.o_T_e

        trajs = []
        assert len(periods) == 3  # period is provided for each step
        # move to the hold position
        trajs.append(self.move_linear_within_time(b_T_ie, periods[0]))

        # # conduct the insertion
        trajs.append(self.move_linear_within_time(b_T_ge, periods[1]))

        # # release the gripper
        # self.release_gripper()

        # # move back to hold position
        trajs.append(self.move_linear_within_time(b_T_ie, periods[2]))
        return trajs

    #### Low-level action ####
    def reset_grasp(self):
        self.gripper.homing()
        self.is_grasping = False

    def grasp(self, grasp_width=GRASP_WIDTH):
        self.gripper.grasp(grasp_width, GRASP_SPEED,
                           GRASP_FORCE, EPS_INNER, EPS_OUTER)
        self.is_grasping = True

    def move_gripper(self, width: float):
        self.gripper.move(width, GRASP_SPEED)

    def release_gripper(self):
        self.gripper.move(RELEASE_WIDTH, GRASP_SPEED)
        self.is_grasping = False

    def move_linear(self, pose: np.ndarray):
        """Robust wrapper of move linear, try multiple times until succeed"""
        if pose.shape == (6,) or pose.shape == (6, 1):
            pos_euler = pose
        elif pose.shape == (4, 4):
            pos_euler = mat2euler(pose)
        elif pose.shape == (3,) or pose.shape == (3, 1):
            # If only position is provided, use TOP_DOWN
            pose_full = np.eye(4, dtype=np.float32)
            pose_full[:3, :3] = TOP_DOWN_R
            pose_full[:3, 3] = pose
            pos_euler = mat2euler(pose_full)
        else:
            raise ValueError("Unknown goal pose shape!")
        for i in range(MAX_TRY):
            try:
                self.robot.move_linear(tuple(pos_euler))
            except Exception as e:
                self.controller.recover_from_errors()
                print(f"Move linear, error: {str(e)} in No.{i} try.")

    def move_joints(self, joints: Tuple):
        """Robust wrapper of move joints, try multiple times until succeed"""
        for i in range(MAX_TRY):
            try:
                self.robot.move_joints(joints)
                break  # Quit after executed
            except Exception as e:
                self.controller.recover_from_errors()
                print(f"Move joint, error: {str(e)} in No.{i} try.")

    def move_linear_velocity(self, velocity: np.ndarray, t: float):
        """Move with linear velocity
        Args:
            - velocity: (x, y, z, wx, wy, wz)
            - t: sec
        """
        for i in range(MAX_TRY):
            try:
                self.robot.controller.move_linear_velocity(velocity)
                time.sleep(t)
                break  # Quit after executed
            except Exception as e:
                self.controller.recover_from_errors()
                print(
                    f"Move linear using velocity, error: {str(e)} in No.{i} try.")

    def move_joint_velocity(self, velocity: np.ndarray, t: float):
        """Move with joint velocity"""
        for i in range(MAX_TRY):
            try:
                self.robot.controller.move_joints_velocity(velocity)
                time.sleep(t)
                break
            except Exception as e:
                self.controller.recover_from_errors()
                print(
                    f"Move joint using velocity, error: {str(e)} in No.{i} try.")

    #### PID Control ####
    def pid_position_control(self, goal_pose):
        """Set goal using pid position control; Convert position goal to velocity control"""
        K_p = 1.0
        K_i = 0.1
        K_d = 0.1
        lin_speed = 0.05  # 5mm per sec
        decrease_region = 0.02  # start to slow when close to 10cm
        t_step = 0.05  # Control in 0.1 sec
        # get current pose
        cur_pose_mat = euler2mat(self.pose)
        # compare with goal pose
        assert goal_pose.shape == (4, 4), "Goal pose should be a mat!"
        goal_pose_mat = goal_pose
        # only for consider linear now
        trans_diff = goal_pose_mat[:3, 3] - cur_pose_mat[:3, 3]  # (3,)
        # smooth when near it using tanh
        # diff compared with speed movement
        diff_scale = np.linalg.norm(trans_diff) / decrease_region
        trans_vel = np.tanh(diff_scale) * trans_diff / \
            np.linalg.norm(trans_diff) * lin_speed
        #
        velocity = np.zeros([6,], dtype=np.float32)
        velocity[:3] = trans_vel
        # Convert back to mm & degree
        velocity[:3] = velocity[:3] * 1000.0
        velocity[3:] = velocity[3:] / np.pi * 180.0
        # self.move_linear_velocity(velocity=velocity, t=t_step)

    def pid_joint_control(self, goal_joint):
        """Set goal using pid position control; Convert position goal to velocity control"""
        K_p = 1.0
        K_i = 0.1
        K_d = 0.1
        t_step = 0.05
        velocity_thresh = 20.0  # (30 degree/s)
        cur_joint = self.robot.joint_angles
        joint_diff = goal_joint - cur_joint
        print(f"Current joint diff: {np.linalg.norm(joint_diff)}")
        velocity = joint_diff / t_step
        # set threshold
        if np.linalg.norm(velocity) > velocity_thresh:
            velocity = velocity / np.linalg.norm(velocity) * velocity_thresh

        decrease_region = 10.0  # start to slow when close to 10cm
        scale = np.tanh(np.linalg.norm(joint_diff) / decrease_region)
        velocity = velocity * scale
        self.move_joint_velocity(velocity=velocity, t=t_step)

    #### Motion generator ####
    def move_linear_within_time(self, goal_pose, period: float, early_stop: bool = True, stop: bool = True):
        """Move to goal pose in period with a constant speed
        Args:
            - early_stop: If we want to early stop if we didn't reach the goal (meaning stop at collision)
        """
        traj = []
        traj_finishsed = True
        start_pose = self.robot.pose
        T_s = euler2mat(start_pose)
        if goal_pose.shape == (6,) or goal_pose.shape == (6, 1):
            T_g = euler2mat(goal_pose)
        elif goal_pose.shape == (4, 4):
            T_g = goal_pose
        else:
            raise ValueError("Unknown goal pose shape!")
        R_s = T_s[:3, :3]
        R_g = T_g[:3, :3]

        # Check the speed limit
        period_lin_min = float(np.linalg.norm(
            T_g[:3, 3] - T_s[:3, 3]) / LINER_SPEED_LIMIT)
        period_ang_min = float(compute_angular_period(
            R_s, R_g, ANGULAR_SPEED_LIMIT))
        period = max(period, max(period_ang_min, period_lin_min))

        num_step = np.floor(period / MIN_TIME_STEP).astype(int)
        linear_vel = (T_g[:3, 3] - T_s[:3, 3]) / period
        angular_vel = compute_angular_velocity(R_s, R_g, period)
        # angular_vel = np.zeros([3,], dtype=np.float32)
        vel = np.concatenate([linear_vel, angular_vel])
        # Convert back to mm & degree
        vel[:3] = vel[:3] * 1000.0
        vel[3:] = vel[3:] / np.pi * 180.0

        system_time = time.time_ns()
        traj.append((system_time, self.robot.pose.tolist()))

        # free_torque = self.controller.ext_torque.copy()
        for i in range(num_step):
            period_i = MIN_TIME_STEP if period >= MIN_TIME_STEP else period
            self.move_linear_velocity(vel, period_i)
            period -= MIN_TIME_STEP
            # record traj
            system_time = time.time_ns()
            traj.append((system_time, self.robot.pose.tolist()))
            # if early_stop:
            #     ext_torque = self.controller.ext_torque
            #     cur_torque = ext_torque.copy()
            #     cur_over_free = np.abs(cur_torque / free_torque)
            #     # print(ext_torque, cur_over_free)
            #     # if np.abs(ext_torque).max() > 3.0 or cur_over_free.max() > 5.0:
            #     if np.abs(ext_torque).max() > 2.5:
            #         traj_finishsed = False
            #         # print(f"Eary stop, pose: {self.robot.pose.tolist()}")
            #         break
        # print(f"Finished, pose: {self.robot.pose.tolist()}")
        # Stop after motion
        if stop:
            self.move_linear_velocity(
                np.zeros(6, dtype=np.float64), MIN_TIME_STEP)
        return traj, traj_finishsed

    def move_towards_linear_within_time(self, goal_pose, period: float, stop: bool = True):
        """Move towards goal pose with a constant speed for period
        """
        traj = []
        traj_finishsed = True
        start_pose = self.robot.pose
        T_s = euler2mat(start_pose)
        if goal_pose.shape == (6,) or goal_pose.shape == (6, 1):
            T_g = euler2mat(goal_pose)
        elif goal_pose.shape == (4, 4):
            T_g = goal_pose
        else:
            raise ValueError("Unknown goal pose shape!")
        R_s = T_s[:3, :3]
        R_g = T_g[:3, :3]

        # Check the speed limit
        num_step = np.floor(period / MIN_TIME_STEP).astype(int)
        linear_vel = (T_g[:3, 3] - T_s[:3, 3]) / period
        angular_vel = compute_angular_velocity(R_s, R_g, period)
        period_ang_min = float(compute_angular_period(
            R_s, R_g, ANGULAR_SPEED_LIMIT))

        # normalize the speed
        if np.linalg.norm(linear_vel) > LINER_SPEED_LIMIT:
            # saturate with max_speed
            linear_vel = linear_vel / \
                np.linalg.norm(linear_vel) * LINER_SPEED_LIMIT
        if period < period_ang_min:
            angular_vel = compute_angular_velocity(R_s, R_g, period_ang_min)

        vel = np.concatenate([linear_vel, angular_vel])
        # Convert back to mm & degree
        vel[:3] = vel[:3] * 1000.0
        vel[3:] = vel[3:] / np.pi * 180.0

        system_time = time.time_ns()
        traj.append((system_time, self.robot.pose.tolist()))

        # free_torque = self.controller.ext_torque.copy()
        for i in range(num_step):
            period_i = MIN_TIME_STEP if period >= MIN_TIME_STEP else period
            self.move_linear_velocity(vel, period_i)
            period -= MIN_TIME_STEP
            # record traj
            system_time = time.time_ns()
            traj.append((system_time, self.robot.pose.tolist()))
            # if early_stop:
            #     ext_torque = self.controller.ext_torque
            #     cur_torque = ext_torque.copy()
            #     cur_over_free = np.abs(cur_torque / free_torque)
            #     # print(ext_torque, cur_over_free)
            #     # if np.abs(ext_torque).max() > 3.0 or cur_over_free.max() > 5.0:
            #     if np.abs(ext_torque).max() > 2.5:
            #         traj_finishsed = False
            #         # print(f"Eary stop, pose: {self.robot.pose.tolist()}")
            #         break
        # print(f"Finished, pose: {self.robot.pose.tolist()}")
        # Stop after motion
        if stop:
            self.move_linear_velocity(
                np.zeros(6, dtype=np.float64), MIN_TIME_STEP)
        return traj, traj_finishsed

    def move_joints_within_time(self, goal_joint, period: float, stop: bool = True):
        """Move to goal joint in period with a constant speed"""
        traj = []
        start_joint = self.robot.joint_angles

        # Check the speed limit
        preiod_min = float(np.linalg.norm(
            goal_joint - start_joint) / JOINT_SPEED_LIMIT)
        preiod = max(period, preiod_min)

        joint_vel = (goal_joint - start_joint) / period

        num_step = np.floor(period / MIN_TIME_STEP).astype(int)
        for i in range(num_step):
            period_i = MIN_TIME_STEP if period >= MIN_TIME_STEP else period
            self.move_joint_velocity(joint_vel, period_i)
            period -= MIN_TIME_STEP
            # record traj
            system_time = time.time_ns()
            traj.append((system_time, self.robot.pose.tolist()))
        # Stop after motion
        if stop:
            self.move_joint_velocity(
                np.zeros(7, dtype=np.float64), MIN_TIME_STEP)
        return traj

    def idle(self):
        """Stop moving"""
        try:
            self.move_linear_velocity(
                np.zeros(6, dtype=np.float64), MIN_TIME_STEP)
        except Exception as e:
            pass
        try:
            self.move_joint_velocity(
                np.zeros(7, dtype=np.float64), MIN_TIME_STEP)
        except Exception as e:
            pass

    #### Property ####
    @property
    def pose(self):
        """Return a (pos (mm), euler (degree)) structure"""
        return self.robot.pose

    @property
    def joint_angles(self):
        return self.robot.joint_angles

    @property
    def flange_pose(self):
        """Return a (pos (m), quat (radius)) structure"""
        return self.robot.controller.flange_pose

    #### Visualization ####
    def show_coordinate(self, pcd, coord_lists: list = []):
        """Show pointcloud, camera coordinate, robot coordinate, and a set of external coordinates
        Args:
            coord_lists: coord_lists should belong to the camera coordinate
        """
        vis_list = []
        # Show point cloud
        vis_list.append(pcd)

        # Show camera coordinate
        camera_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.02)
        camera_origin.paint_uniform_color([1.0, 0.0, 0.0])
        vis_list.append(camera_origin)

        # Show robot coordinate
        self.update_tf()
        robot_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1)
        robot_origin.transform(self.c_T_b)
        vis_list.append(robot_origin)

        # Draw a plane
        plane = o3d_create_plane()
        plane.transform(self.c_T_b)
        vis_list.append(plane)

        # Show external coordinate
        for coord in coord_lists:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.02)
            origin.transform(coord)
            vis_list.append(origin)
        o3d.visualization.draw_geometries(vis_list)
