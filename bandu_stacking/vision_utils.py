import copy
import os
from collections import OrderedDict

import numpy as np
import torch

import bandu_stacking.pb_utils as pbu
from bandu_stacking.ucn.lib.fcn.config import cfg, cfg_from_file
from bandu_stacking.ucn.lib.fcn.test_dataset import test_sample
from bandu_stacking.ucn.lib.networks.SEG import seg_resnet34_8s_embedding

FLOOR = 0
TABLE_IDNUM = 1
BACKGROUND = [FLOOR, TABLE_IDNUM]

CFG_PATH = "experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml"
SAMPLING_CKPT_PATH = "data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth"
CROP_CKPT_PATH = "data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth"


UNKNOWN = "unknown"
TABLE = "table"
SPECIAL_CATEGORIES = {None: pbu.BLACK, UNKNOWN: pbu.GREY, TABLE: pbu.WHITE}


def image_from_labeled(seg_image, **kwargs):

    # TODO: order special colors
    # TODO: adjust saturation and value per category
    # labels = sorted(set(get_bodies()) | set(seg_image[..., 0].flatten()))
    labels_instance = set(seg_image[..., 1].flatten())
    detect_obj_labels = sorted(
        label
        for label in labels_instance
        if isinstance(label, str) and (label not in SPECIAL_CATEGORIES)
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


def cloud_from_depth(camera_matrix, depth, max_depth=10.0, top_left_origin=False):
    # width, height = map(int, dimensions_from_camera_matrix(camera_matrix))
    height, width = depth.shape
    xmap = np.array(
        [[i for i in range(width)] for _ in range(height)]
    )  # 0 ~ width. hxw
    if top_left_origin:
        ymap = np.array(
            [[j for _ in range(width)] for j in range(height)]
        )  # 0 ~ height. hxw
    else:
        ymap = np.array(
            [[height - j for _ in range(width)] for j in range(height)]
        )  # 0 ~ height. hxw
    homogeneous_coord = np.concatenate(
        [xmap.reshape(1, -1), ymap.reshape(1, -1), np.ones((1, height * width))]
    )  # 3 x (hw)

    rays = np.linalg.inv(camera_matrix).dot(homogeneous_coord)
    point_cloud = depth.reshape(1, height * width) * rays
    point_cloud = point_cloud.transpose(1, 0).reshape(height, width, 3)

    # Filter max depth
    point_cloud[point_cloud[:, :, 2] > max_depth] = 0
    return point_cloud


def fuse_predicted_labels(
    seg_network, camera_image: pbu.CameraImage, use_depth=False, num_segs=1, **kwargs
):

    point_cloud = None
    if use_depth:
        point_cloud = cloud_from_depth(
            camera_image.camera_matrix, camera_image.depthPixels
        )

    predicted_seg = seg_network.get_seg(
        camera_image.rgbPixels[:, :, :3],
        point_cloud=point_cloud,
        depth_image=camera_image.depthPixels,
        return_int=False,
        num_segs=num_segs,
        **kwargs,
    )
    new_camera_image = copy.deepcopy(camera_image)
    new_camera_image.segmentationMaskBuffer = predicted_seg
    return new_camera_image


def save_camera_images(
    camera_image: pbu.CameraImage, directory="./logs", prefix="", **kwargs
):
    # safe_remove(directory)
    pbu.ensure_dir(directory)
    # depth_image = simulate_depth(depth_image)
    pbu.save_image(
        os.path.join(directory, "{}rgb.png".format(prefix)), camera_image.rgbPixels
    )  # [0, 255]

    pbu.save_image(
        os.path.join(directory, "{}depth.png".format(prefix)), camera_image.depthPixels
    )  # [0, 1]

    if camera_image.segmentationMaskBuffer is None:
        return None

    segmented_image = image_from_labeled(camera_image.segmentationMaskBuffer, **kwargs)

    pbu.save_image(
        os.path.join(directory, "{}segmented.png".format(prefix)), segmented_image
    )  # [0, 255]
    return segmented_image


class CategoryAgnosticSeg(object):
    """UCN or UOIS."""

    # common stat from ImageNet
    image_net_mean = [0.485, 0.456, 0.406]
    image_net_std = [0.229, 0.224, 0.225]

    def __init__(self):
        pass

    def erode(self, masks, kernel_size=3):
        from scipy.ndimage import binary_erosion

        for i in np.unique(masks):
            if i in BACKGROUND:
                continue
            mask = masks == i
            boundary_mask = np.logical_xor(
                binary_erosion(mask, structure=np.ones((kernel_size, kernel_size))),
                mask,
            )
            masks[boundary_mask] = TABLE_IDNUM
        return masks


class UCN(CategoryAgnosticSeg):
    def __init__(self, base_path, device="cuda", **kwargs):
        super().__init__()
        self.device = device
        self.base_path = base_path

        self.config_path = os.path.join(self.base_path, CFG_PATH)

        cfg_from_file(self.config_path)
        cfg.device = self.device
        cfg.instance_id = 0
        num_classes = 2
        cfg.MODE = "TEST"
        cfg.TEST.VISUALIZE = False

        sampling_path = os.path.join(self.base_path, SAMPLING_CKPT_PATH)
        network_data = torch.load(sampling_path, map_location=cfg.device)
        self.network = seg_resnet34_8s_embedding(
            num_classes, cfg.TRAIN.NUM_UNITS, network_data
        ).to(device=cfg.device)
        self.network.eval()

        crop_path = os.path.join(self.base_path, CROP_CKPT_PATH)
        network_data_crop = torch.load(crop_path, map_location=cfg.device)
        self.network_crop = seg_resnet34_8s_embedding(
            num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop
        ).to(device=cfg.device)
        self.network_crop.eval()

    def seg_fn(self, x):
        return test_sample(x, self.network, self.network_crop)

    def get_seg(self, rgb_image, point_cloud=None, **kwargs):

        # NOTE. ori input and output - 1) y axis pointing downward. 2) 0 for bg, 1+ for fg
        image_standardized = np.zeros_like(rgb_image).astype(np.float32)
        for i in range(3):
            image_standardized[..., i] = (
                rgb_image[..., i] / 255.0 - self.image_net_mean[i]
            ) / self.image_net_std[i]
        image_standardized = image_standardized[..., ::-1].copy()
        # im in bgr order
        point_cloud[..., 1] *= -1  # y axis pointing downward. (reversed in UOIS)
        batch = {
            "image_color": torch.from_numpy(image_standardized)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .to(self.device),
            "depth": torch.from_numpy(point_cloud)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
            .to(self.device),
        }

        _, instance_mask = self.seg_fn(batch)

        if instance_mask is not None:
            instance_mask = instance_mask[0].detach().cpu().numpy()
        else:
            instance_mask = np.zeros(rgb_image.shape[:2])
        instance_mask[instance_mask == 1] = instance_mask.max() + 1

        instances = np.unique(instance_mask)
        instances = instances[~np.isin(instances, BACKGROUND)]
        segment = np.zeros(rgb_image.shape[:2] + (2,), dtype=int)  # H x W x 2

        segment[np.isin(instance_mask, BACKGROUND), 0] = FLOOR

        return segment
