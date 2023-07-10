import os
from dataclasses import dataclass, field
from typing import Dict

from effdet.data.dataset_config import *



# FLIR-Aligned Dataset
@dataclass
class FlirAlignedFullCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/thermal/flir_train.json', img_dir='images_thermal_train/data/', has_labels=True),
        val=dict(ann_filename='meta/thermal/flir_test.json', img_dir='images_thermal_test/data/', has_labels=True),
        test=dict(ann_filename='meta/thermal/flir_test.json', img_dir='images_thermal_test/data/', has_labels=True),
    ))


@dataclass
class FlirAlignedDayCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/thermal/day_flir_train.json', img_dir='images_thermal_train/data/', has_labels=True),
        val=dict(ann_filename='meta/thermal/day_flir_test.json', img_dir='images_thermal_test/data/', has_labels=True),
        test=dict(ann_filename='meta/thermal/day_flir_test.json', img_dir='images_thermal_test/data/', has_labels=True),
    ))



@dataclass
class FlirAlignedNightCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/thermal/night_flir_train.json', img_dir='images_thermal_train/data/', has_labels=True),
        val=dict(ann_filename='meta/thermal/night_flir_test.json', img_dir='images_thermal_test/data/', has_labels=True),
        test=dict(ann_filename='meta/thermal/night_flir_test.json', img_dir='images_thermal_test/data/', has_labels=True),
    ))