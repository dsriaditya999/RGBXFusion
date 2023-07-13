""" COCO transforms (quick and dirty)

Hacked together by Ross Wightman
"""
import random
import math
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
import cv2

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

FLIR_THERMAL_MEAN = (0.519, 0.519, 0.519)
FLIR_THERMAL_STD = (0.225, 0.225, 0.225)

class ImageToNumpy:

    def __call__(self, np_img, annotations: dict):
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img, annotations
    

class ImageToNumpy:

    def __call__(self, pil_gated_img, pil_rgb_img, annotations: dict):
        np_gated_img = np.array(pil_gated_img, dtype=np.uint8)
        np_rgb_img = np.array(pil_rgb_img, dtype=np.uint8)
        if np_gated_img.ndim < 3:
            np_gated_img = np.expand_dims(np_gated_img, axis=-1)
        np_gated_img = np.moveaxis(np_gated_img, 2, 0)  # HWC to CHW
        if np_rgb_img.ndim < 3:
            np_rgb_img = np.expand_dims(np_rgb_img, axis=-1)
        np_rgb_img = np.moveaxis(np_rgb_img, 2, 0)  # HWC to CHW
        return np_gated_img, np_rgb_img, annotations

def cv2_interpolation(method):
    if method == 'bicubic':
        return cv2.INTER_CUBIC
    elif method == 'lanczos':
        return cv2.INTER_LANCZOS4
    else:
        # default bilinear, do we want to allow nearest?
        return cv2.INTER_LINEAR


_RANDOM_INTERPOLATION = (cv2.INTER_LINEAR, cv2.INTER_CUBIC)


def clip_boxes_(boxes, img_size):
    height, width = img_size
    clip_upper = np.array([height, width] * 2, dtype=boxes.dtype)
    np.clip(boxes, 0, clip_upper, out=boxes)


def clip_boxes(boxes, img_size):
    clipped_boxes = boxes.copy()
    clip_boxes_(clipped_boxes, img_size)
    return clipped_boxes


def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size


class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear', rgb_fill_color: tuple = (0, 0, 0), gated_fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.rgb_fill_color = rgb_fill_color
        self.gated_fill_color = gated_fill_color

    def __call__(self, gated_img, rgb_img, anno: dict):
        height, width = gated_img.shape[:2]

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        interp_method = cv2_interpolation(self.interpolation)


        new_gated_img = np.zeros((self.target_size[1], self.target_size[0], 3))
        gated_img = cv2.resize(gated_img, (scaled_w, scaled_h), interpolation=interp_method)
        new_gated_img[:scaled_h,:scaled_w] = gated_img

        new_rgb_img = np.zeros((self.target_size[1], self.target_size[0], 3))
        rgb_img = cv2.resize(rgb_img, (scaled_w, scaled_h), interpolation=interp_method)
        new_rgb_img[:scaled_h,:scaled_w] = rgb_img

        if 'bbox' in anno:
            bbox = anno['bbox']
            bbox[:, :4] *= img_scale
            bbox_bound = (min(scaled_h, self.target_size[0]), min(scaled_w, self.target_size[1]))
            clip_boxes_(bbox, bbox_bound)  # crop to bounds of target image or letter-box, whichever is smaller
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        anno['img_scale'] = 1. / img_scale  # back to original

        return new_gated_img, new_rgb_img, anno


class RandomResizePad:

    def __init__(self, target_size: int, scale: tuple = (0.5, 2.0), interpolation: str = 'random',
                 rgb_fill_color: tuple = (0, 0, 0), gated_fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.scale = scale
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = cv2_interpolation(interpolation)
        self.rgb_fill_color = rgb_fill_color
        self.gated_fill_color = gated_fill_color

    def _get_params(self, img):
        # Select a random scale factor.
        scale_factor = random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]

        # Recompute the accurate scale_factor using rounded scaled image size.
        height, width = img.shape[:2]
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * random.uniform(0, 1))
        return scaled_h, scaled_w, offset_y, offset_x, img_scale

    def __call__(self, gated_img, rgb_img, anno: dict):
        scaled_h, scaled_w, offset_y, offset_x, img_scale = self._get_params(gated_img)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        
        right, lower = min(scaled_w, offset_x + self.target_size[1]), min(scaled_h, offset_y + self.target_size[0])

        gated_img = cv2.resize(gated_img, (scaled_w, scaled_h), interpolation=interpolation)
        gated_img = gated_img[offset_y:lower, offset_x:right]
        new_gated_img = np.zeros((self.target_size[1], self.target_size[0], 3))
        h, w, c = gated_img.shape
        new_gated_img[:h,:w] = gated_img

        rgb_img = cv2.resize(rgb_img, (scaled_w, scaled_h), interpolation=interpolation)
        rgb_img = rgb_img[offset_y:lower, offset_x:right]
        new_rgb_img = np.zeros((self.target_size[1], self.target_size[0], 3))
        h, w, c = rgb_img.shape
        new_rgb_img[:h,:w] = rgb_img

        if 'bbox' in anno:
            bbox = anno['bbox']  # for convenience, modifies in-place
            bbox[:, :4] *= img_scale
            box_offset = np.stack([offset_y, offset_x] * 2)
            bbox -= box_offset
            bbox_bound = (min(scaled_h, self.target_size[0]), min(scaled_w, self.target_size[1]))
            clip_boxes_(bbox, bbox_bound)  # crop to bounds of target image or letter-box, whichever is smaller
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        anno['img_scale'] = 1. / img_scale  # back to original

        return new_gated_img, new_rgb_img, anno


class RandomFlip:

    def __init__(self, horizontal=True, vertical=False, prob=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.prob = prob

    def _get_params(self):
        do_horizontal = random.random() < self.prob if self.horizontal else False
        do_vertical = random.random() < self.prob if self.vertical else False
        return do_horizontal, do_vertical

    def __call__(self, gated_img, rgb_img, annotations: dict):
        do_horizontal, do_vertical = self._get_params()
        height, width = gated_img.shape[:2]

        def _fliph(bbox):
            x_max = width - bbox[:, 1]
            x_min = width - bbox[:, 3]
            bbox[:, 1] = x_min
            bbox[:, 3] = x_max

        def _flipv(bbox):
            y_max = height - bbox[:, 0]
            y_min = height - bbox[:, 2]
            bbox[:, 0] = y_min
            bbox[:, 2] = y_max

        if do_horizontal and do_vertical:
            gated_img = cv2.rotate(gated_img, cv2.ROTATE_180)
            rgb_img = cv2.rotate(rgb_img, cv2.ROTATE_180)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
                _flipv(annotations['bbox'])
        elif do_horizontal:
            gated_img = cv2.flip(gated_img, 1)
            rgb_img = cv2.flip(rgb_img, 1)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
        elif do_vertical:
            gated_img = cv2.flip(gated_img, 0)
            rgb_img = cv2.flip(rgb_img, 0)
            if 'bbox' in annotations:
                _flipv(annotations['bbox'])

        return gated_img, rgb_img, annotations


def resolve_fill_color(fill_color, img_mean=IMAGENET_DEFAULT_MEAN):
    if isinstance(fill_color, tuple):
        assert len(fill_color) == 3
        fill_color = fill_color
    else:
        try:
            int_color = int(fill_color)
            fill_color = (int_color,) * 3
        except ValueError:
            assert fill_color == 'mean'
            fill_color = tuple([int(round(255 * x)) for x in img_mean])
    return fill_color

    

class Compose:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, gated_img, rgb_img, annotations: dict):
        for t in self.transforms:
            gated_img, rgb_img, annotations = t(gated_img, rgb_img, annotations)
        return gated_img, rgb_img, annotations


def transforms_coco_eval(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        fill_color='mean',
        rgb_mean=IMAGENET_DEFAULT_MEAN,
        rgb_std=IMAGENET_DEFAULT_STD,
        gated_mean=IMAGENET_DEFAULT_MEAN,
        gated_std=IMAGENET_DEFAULT_STD):

    rgb_fill_color, gated_fill_color = resolve_fill_color(fill_color, rgb_mean), resolve_fill_color(fill_color, gated_mean)

    image_tfl = [
        ResizePad(
            target_size=img_size, interpolation=interpolation, rgb_fill_color=rgb_fill_color, gated_fill_color=gated_fill_color),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf


def transforms_coco_train(
        img_size=224,
        interpolation='random',
        use_prefetcher=False,
        fill_color='mean',
        rgb_mean=IMAGENET_DEFAULT_MEAN,
        rgb_std=IMAGENET_DEFAULT_STD,
        gated_mean=IMAGENET_DEFAULT_MEAN,
        gated_std=IMAGENET_DEFAULT_STD):

    rgb_fill_color, gated_fill_color = resolve_fill_color(fill_color, rgb_mean), resolve_fill_color(fill_color, gated_mean)

    image_tfl = [
        RandomFlip(horizontal=True, prob=0.5),
        RandomResizePad(
            target_size=img_size, interpolation=interpolation, rgb_fill_color=rgb_fill_color, gated_fill_color=gated_fill_color),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf
