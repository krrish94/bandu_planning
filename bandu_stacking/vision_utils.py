from __future__ import print_function

import copy
import os
from collections import OrderedDict

import numpy as np
import torch

import bandu_stacking.pb_utils as pbu
from bandu_stacking.ucn.lib.fcn.config import cfg, cfg_from_file
from bandu_stacking.ucn.lib.fcn.test_dataset import test_sample
from bandu_stacking.ucn.lib.networks.SEG import seg_resnet34_8s_embedding
import matplotlib.pyplot as plt

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

    print(camera_image.depthPixels.shape)
    predicted_seg = seg_network.get_seg(
        camera_image.rgbPixels[:, :, :3],
        camera_image.depthPixels,
        camera_image.camera_matrix,
        **kwargs,
    )
    new_camera_image = copy.deepcopy(camera_image)
    new_camera_image.segmentationMaskBuffer = predicted_seg
    return new_camera_image


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


def build_matrix_of_indices(height, width):
    """Builds a [height, width, 2] numpy array containing coordinates.

    @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)

def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = build_matrix_of_indices(height, width)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


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

def get_seg_sam(image):
    import sys
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    checkpoint_path = os.path.abspath(os.path.join(__file__, *[os.pardir] * 2, "checkpoints"))
    sam_checkpoint = os.path.join(checkpoint_path, "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

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

    def get_seg(self, rgb_image, depth_image, intrinsics, **kwargs):
        
     
        height = rgb_image.shape[0]
        width = rgb_image.shape[1]

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        px = intrinsics[0, 2]
        py = intrinsics[1, 2]

        # bgr image
        im = rgb_image.astype(np.float32)
        im = im[:, :, (2, 1, 0)]

        # xyz image
        xyz_img = compute_xyz(depth_image, fx, fy, px, py, height, width)

        im_tensor = torch.from_numpy(im) / 255.0
        pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        im_tensor -= pixel_mean
        image_blob = im_tensor.permute(2, 0, 1)
        sample = {"image_color": image_blob.unsqueeze(0)}

        if cfg.INPUT == "DEPTH" or cfg.INPUT == "RGBD":
            depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
            sample["depth"] = depth_blob.unsqueeze(0).type(torch.cuda.FloatTensor)

        _, instance_mask = self.seg_fn(sample)

        if instance_mask is not None:
            instance_mask = instance_mask[0].detach().cpu().numpy()
        else:
            instance_mask = np.zeros(rgb_image.shape[:2])
        instance_mask[instance_mask == 1] = instance_mask.max() + 1

        instances = np.unique(instance_mask)
        instances = instances[~np.isin(instances, BACKGROUND)]
        segment = np.zeros(rgb_image.shape[:2] + (2,), dtype=int)  # H x W x 2
        
        for i in instances:
            segment[instance_mask == i, 0] = np.iinfo(int).max 
            segment[instance_mask == i, 1] = i

        segment[np.isin(instance_mask, BACKGROUND), 0] = FLOOR

        return segment
