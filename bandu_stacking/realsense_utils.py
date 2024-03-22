
import os
import pyrealsense2 as rs



import os
import os.path as osp
import json
from enum import IntEnum
from types import SimpleNamespace
import sys
sys.path.append(osp.abspath(__file__))

import cv2
import numpy as np

import pyrealsense2 as rs

# from helper.utils import make_dir

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

'''
Query for the color and depth stream profiles available with the Intel Realsense camera
DEBUG this code to get the appropriate color and depth profile
'''
def get_profiles():
    ctx = rs.context()
    devices = ctx.query_devices()

    color_profiles = []
    depth_profiles = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f'Sensor: {name}, {serial}')
        print('Supported video formats: ')
        for sensor in device.query_sensors():
            for stream_profile in sensor.get_stream_profiles():
                stream_type = str(stream_profile.stream_type())

                if stream_type in ['stream.color', 'stream.depth']:
                    v_profile = stream_profile.as_video_stream_profile()
                    fmt = stream_profile.format()
                    w, h = v_profile.width(), v_profile.height()
                    fps = v_profile.fps()

                    video_type = stream_type.split('.')[-1]
                    print(f'Video type: {video_type}, width={w}, height={h}, fps={fps}, format={fmt}')
                    if video_type == 'color':
                        color_profiles.append((w, h, fps, fmt))
                    else:
                        depth_profiles.append((w, h, fps, fmt))

    return color_profiles, depth_profiles

'''
Capture the scene from a given pose (pose_index)
'''
def realsense_capture(pipeline, depth_sensor, align):

    # Streaming loop
    print(f'Depth preset value : {depth_sensor.get_option(rs.option.visual_preset)}')

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames(timeout_ms=10000)

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    intrinsics = frames.profile.as_video_stream_profile().intrinsics

    return depth_image, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), intrinsics


    
def capture_realsense_image_by_serial_number(output_folder, serial_number):

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