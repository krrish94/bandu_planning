'''
This code is used to capture RGB, Depth, and Camera intrinsics using the Intel Realsense camera 
Default modes specified for camera -> Custom, Default, Hand, HighAccuracy, HighDensity, MediumDensity -- for tabletop manipulation, we observe that Default (Enum -> 1) setting gives us the best capture
'''

import os
import os.path as osp
import json
from enum import IntEnum

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

def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)

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
def realsense_capture(args, pipeline, depth_sensor, align, pose_index, frames_to_skip=0):
    frames_to_capture = args.frames_to_capture

    path_output = args.output_folder
    path_depth = osp.join(args.output_folder, f'{pose_index}', "depth")
    path_color = osp.join(args.output_folder, f'{pose_index}', "rgb")
    print(path_depth, args.record_images)

    if args.record_images:
        os.makedirs(path_depth, exist_ok=True)
        os.makedirs(path_color, exist_ok=True)
        # make_dir(path_depth)
        # make_dir(path_color)

    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    # clipping_distance_in_meters meters away
    clipping_distance_in_meters = args.clipping_distance  # 3 meter --> ADD YOUR OWN CLIPPING DISTANCE HERE (3 meters is good for manipulator experiments)
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Streaming loop
    frame_count = 0
    print(f'Capturing {frames_to_capture} frames')
    try:
        while True and (frame_count < frames_to_capture or frame_count < 0):

            print(f'Depth preset value : {depth_sensor.get_option(rs.option.visual_preset)}')
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames(timeout_ms=10000)

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if frames_to_skip > 1:
                frames_to_skip = frames_to_skip - 1
                continue

            if args.record_images:
                if frame_count == 0:
                    save_intrinsic_as_json(osp.join(path_output, f'{pose_index}', "camera_intrinsic.json"), color_frame)

                cv2.imwrite(f"{path_depth}/{str(frame_count).zfill(5)}.png", depth_image)
                cv2.imwrite(f"{path_color}/{str(frame_count).zfill(5)}.jpg", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
                print(f"Saved color and depth image {str(frame_count).zfill(5)}")
                frame_count += 1
            else:
                frame_count = -1

            # # Remove background - Set pixels further than clipping_distance to grey
            # grey_color = 153
            # # Depth image is 1 channel, color is 3 channels
            # depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            
            # Render images
            if args.render_images:
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
                images = np.hstack((bg_removed, depth_colormap))
                cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Recorder Realsense', images)
                key = cv2.waitKey(1)

                # if 'esc' button pressed, escape loop and exit program
                if key == 27:
                    cv2.destroyAllWindows()
                    break
    finally:
        pass
        # pipeline.stop()