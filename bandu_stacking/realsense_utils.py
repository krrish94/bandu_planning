from __future__ import print_function

import os
from enum import IntEnum
from types import SimpleNamespace

import numpy as np

from bandu_stacking.pb_utils import CameraImage

CALIB_DIR = os.path.join(
    os.path.dirname(__file__), "../calibration/current_calibration/calib"
)
CAMERA_SNS = [
    # "103422071983",
    "027322071284",
    "050522073498",
    # "102422072672",
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
    import pyrealsense2 as rs
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

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    if serial_number is not None:
        config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    pipeline_profile = pipeline.start(config)

    for _ in range(100):
        # Wait for a coherent pair of frames: depth and color
        frameset = pipeline.wait_for_frames()

    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    depth = np.asanyarray(aligned_depth_frame.get_data())
    rgb = np.asanyarray(frameset.get_color_frame().get_data())

    # And get the device info
    print(f"Connected to {serial_number}")

    intrinsics, _ = get_intrinsics(pipeline_profile, rs.stream.color)

    return CameraImage(rgb, depth / 1000.0, None, camera_pose, intrinsics)
