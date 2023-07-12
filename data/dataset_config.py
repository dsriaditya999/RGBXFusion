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

# M3FD Dataset
@dataclass
class M3fdDayCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-daytime-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-daytime-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-daytime-test.json', img_dir='Ir', has_labels=True)
    ))


@dataclass
class M3fdNightCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-night-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-night-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-night-test.json', img_dir='Ir', has_labels=True)
    ))


@dataclass
class M3fdOvercastCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-overcast-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-overcast-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-overcast-test.json', img_dir='Ir', has_labels=True)
    ))

@dataclass
class M3fdChallengeCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-challenge-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-challenge-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-challenge-test.json', img_dir='Ir', has_labels=True)
    ))

@dataclass
class M3fdFullCfg(CocoCfg):
    variant: str = ''
    splits: Dict[str, dict] = field(default_factory=lambda: dict(
        train=dict(ann_filename='meta/m3fd-train.json', img_dir='Ir', has_labels=True),
        val=dict(ann_filename='meta/m3fd-val.json', img_dir='Ir', has_labels=True),
        test=dict(ann_filename='meta/m3fd-test.json', img_dir='Ir', has_labels=True)
    ))