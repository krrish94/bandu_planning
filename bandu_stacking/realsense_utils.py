from __future__ import annotations

import os
from enum import IntEnum
from types import SimpleNamespace

import numpy as np

from bandu_stacking.pb_utils import CameraImage

CALIB_DIR = os.path.join(
    os.path.dirname(__file__), "../calibration/current_calibration/calib"
)
CAMERA_SNS = [
    "103422071983",
    "027322071284",
    "050522073498",
    "102422072672",
]


def rs_intrinsics_to_opencv_intrinsics(intr):
    D = np.array(intr.coeffs)
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    return K, D


def get_intrinsics(pipeline_profile, stream):
    stream_profile = pipeline_profile.get_stream(
        stream
    )  # Fetch stream profile for depth stream
    intr = (
        stream_profile.as_video_stream_profile().get_intrinsics()
    )  # Downcast to video_stream_profile and fetch intrinsics
    return rs_intrinsics_to_opencv_intrinsics(intr)


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


"""
Query for the color and depth stream profiles available with the Intel Realsense camera
DEBUG this code to get the appropriate color and depth profile
"""


def get_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()

    color_profiles = []
    depth_profiles = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"Sensor: {name}, {serial}")
        print("Supported video formats: ")
        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ["stream.color", "stream.depth"]:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()

                    video_type = stream_type.split(".")[-1]
                    print(
                        f"Video type: {video_type}, width={w}, height={h}, fps={fps}, format={fmt}"
                    )
                    if video_type == "color":
                        color_profiles.append((w, h, fps, fmt))
                    else:
                        depth_profiles.append((w, h, fps, fmt))

    return color_profiles, depth_profiles


"""
Capture the scene from a given pose (pose_index)
"""


def get_camera_image(serial_number, camera_pose):
    import pyrealsense2 as rs

    rs_args = SimpleNamespace()
    rs_args.realsense_preset = 1
    rs_args.clipping_distance = 3
    rs_args.frames_to_capture = 1
    rs_args.render_images = False
    rs_args.record_images = True
    rs_args.color_profile = 14  # 42
    rs_args.depth_profile = 5

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

    rgb, depth, intrinsics = realsense_capture(pipeline, profile, depth_sensor, align)
    return CameraImage(rgb, depth / 1000.0, None, camera_pose, intrinsics)


def realsense_capture(pipeline, profile, depth_sensor, align):
    import pyrealsense2 as rs

    # Streaming loop
    print(f"Depth preset value : {depth_sensor.get_option(rs.option.visual_preset)}")

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames(timeout_ms=10000)

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    intrinsics, _ = get_intrinsics(profile, rs.stream.color)

    return color_image, depth_image, intrinsics
