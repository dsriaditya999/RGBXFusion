""" Flir transforms
"""
import random

from PIL import Image
from PIL import ImageEnhance
import numpy as np

from effdet.data.transforms import _pil_interp, _size_tuple, clip_boxes_
from effdet.data.transforms import resolve_fill_color

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


class ImageToNumpy:

    def __call__(self, pil_thermal_img, pil_rgb_img, annotations: dict):
        np_thermal_img = np.array(pil_thermal_img, dtype=np.uint8)
        np_rgb_img = np.array(pil_rgb_img, dtype=np.uint8)
        if np_thermal_img.ndim < 3:
            np_thermal_img = np.expand_dims(np_thermal_img, axis=-1)
        np_thermal_img = np.moveaxis(np_thermal_img, 2, 0)  # HWC to CHW
        if np_rgb_img.ndim < 3:
            np_rgb_img = np.expand_dims(np_rgb_img, axis=-1)
        np_rgb_img = np.moveaxis(np_rgb_img, 2, 0)  # HWC to CHW
        return np_thermal_img, np_rgb_img, annotations


class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear', rgb_fill_color: tuple = (0, 0, 0), thermal_fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.rgb_fill_color = rgb_fill_color
        self.thermal_fill_color = thermal_fill_color

    def __call__(self, thermal_img, rgb_img, anno: dict):
        width, height = thermal_img.size

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        interp_method = _pil_interp(self.interpolation)
        thermal_img = thermal_img.resize((scaled_w, scaled_h), interp_method)
        new_thermal_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.rgb_fill_color)
        new_thermal_img.paste(thermal_img)  # pastes at 0,0 (upper-left corner)

        rgb_img = rgb_img.resize((scaled_w, scaled_h), interp_method)
        new_rgb_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.thermal_fill_color)
        new_rgb_img.paste(rgb_img)  # pastes at 0,0 (upper-left corner)

        if 'bbox' in anno:
            bbox = anno['bbox']
            bbox[:, :4] *= img_scale
            bbox_bound = (min(scaled_h, self.target_size[0]), min(scaled_w, self.target_size[1]))
            clip_boxes_(bbox, bbox_bound)  # crop to bounds of target image or letter-box, whichever is smaller
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        anno['img_scale'] = 1. / img_scale  # back to original

        return new_thermal_img, new_rgb_img, anno


class RandomResizePad:

    def __init__(self, target_size: int, scale: tuple = (0.1, 2.0), interpolation: str = 'random',
                 rgb_fill_color: tuple = (0, 0, 0), thermal_fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.scale = scale
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.rgb_fill_color = rgb_fill_color
        self.thermal_fill_color = thermal_fill_color

    def _get_params(self, img):
        # Select a random scale factor.
        scale_factor = random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]

        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.size
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

    def __call__(self, thermal_img, rgb_img, anno: dict):
        scaled_h, scaled_w, offset_y, offset_x, img_scale = self._get_params(thermal_img)

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        right, lower = min(scaled_w, offset_x + self.target_size[1]), min(scaled_h, offset_y + self.target_size[0])
        thermal_img = thermal_img.resize((scaled_w, scaled_h), interpolation)
        thermal_img = thermal_img.crop((offset_x, offset_y, right, lower))
        new_thermal_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.thermal_fill_color)
        new_thermal_img.paste(thermal_img)  # pastes at 0,0 (upper-left corner)

        rgb_img = rgb_img.resize((scaled_w, scaled_h), interpolation)
        rgb_img = rgb_img.crop((offset_x, offset_y, right, lower))
        new_rgb_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.rgb_fill_color)
        new_rgb_img.paste(rgb_img)  # pastes at 0,0 (upper-left corner)

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

        return new_thermal_img, new_rgb_img, anno


class RandomFlip:

    def __init__(self, horizontal=True, vertical=False, prob=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.prob = prob

    def _get_params(self):
        do_horizontal = random.random() < self.prob if self.horizontal else False
        do_vertical = random.random() < self.prob if self.vertical else False
        return do_horizontal, do_vertical

    def __call__(self, thermal_img, rgb_img, annotations: dict):
        do_horizontal, do_vertical = self._get_params()
        width, height = thermal_img.size

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
            thermal_img = thermal_img.transpose(Image.ROTATE_180)
            rgb_img = rgb_img.transpose(Image.ROTATE_180)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
                _flipv(annotations['bbox'])
        elif do_horizontal:
            thermal_img = thermal_img.transpose(Image.FLIP_LEFT_RIGHT)
            rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
        elif do_vertical:
            thermal_img = thermal_img.transpose(Image.FLIP_TOP_BOTTOM)
            rgb_img = rgb_img.transpose(Image.FLIP_TOP_BOTTOM)
            if 'bbox' in annotations:
                _flipv(annotations['bbox'])

        return thermal_img, rgb_img, annotations


class RandomColorJitter:

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, sharpness=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.sharpness = sharpness


    def _get_params(self):

        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        sharpness_factor = random.uniform(max(0, 1 - self.sharpness), 1 + self.sharpness)

        return brightness_factor, contrast_factor, saturation_factor, sharpness_factor

    def __call__(self, thermal_img, rgb_img, annotations: dict):

        brightness_factor, contrast_factor, saturation_factor, sharpness_factor = self._get_params()

        thermal_img = ImageEnhance.Brightness(thermal_img).enhance(brightness_factor)
        thermal_img = ImageEnhance.Contrast(thermal_img).enhance(contrast_factor)
        thermal_img = ImageEnhance.Sharpness(thermal_img).enhance(sharpness_factor)

        brightness_factor, contrast_factor, saturation_factor, sharpness_factor = self._get_params()

        rgb_img = ImageEnhance.Brightness(rgb_img).enhance(brightness_factor)
        rgb_img = ImageEnhance.Contrast(rgb_img).enhance(contrast_factor)
        rgb_img = ImageEnhance.Color(rgb_img).enhance(saturation_factor)
        rgb_img = ImageEnhance.Sharpness(rgb_img).enhance(sharpness_factor)

        return thermal_img, rgb_img, annotations



class Compose:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, thermal_img, rgb_img, annotations: dict):
        for t in self.transforms:
            thermal_img, rgb_img, annotations = t(thermal_img, rgb_img, annotations)
        return thermal_img, rgb_img, annotations


def transforms_eval(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        fill_color='mean',
        rgb_mean=IMAGENET_DEFAULT_MEAN,
        rgb_std=IMAGENET_DEFAULT_STD,
        thermal_mean=IMAGENET_DEFAULT_MEAN,
        thermal_std=IMAGENET_DEFAULT_STD):

    rgb_fill_color, thermal_fill_color = resolve_fill_color(fill_color, rgb_mean), resolve_fill_color(fill_color, thermal_mean)

    image_tfl = [
        ResizePad(
            target_size=img_size, interpolation=interpolation, rgb_fill_color=rgb_fill_color, thermal_fill_color=thermal_fill_color),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf


def transforms_train(
        img_size=224,
        interpolation='random',
        use_prefetcher=False,
        fill_color='mean',
        rgb_mean=IMAGENET_DEFAULT_MEAN,
        rgb_std=IMAGENET_DEFAULT_STD,
        thermal_mean=IMAGENET_DEFAULT_MEAN,
        thermal_std=IMAGENET_DEFAULT_STD):

    rgb_fill_color, thermal_fill_color = resolve_fill_color(fill_color, rgb_mean), resolve_fill_color(fill_color, thermal_mean)

    image_tfl = [
        RandomFlip(horizontal=True, prob=0.5),
        RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, sharpness=0.4),
        RandomResizePad(
            target_size=img_size, interpolation=interpolation, rgb_fill_color=rgb_fill_color, thermal_fill_color=thermal_fill_color),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf
