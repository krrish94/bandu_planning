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
