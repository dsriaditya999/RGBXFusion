""" Dataset factory
"""
from collections import OrderedDict
from pathlib import Path

from effdet.data.parsers import *
from effdet.data.parsers import create_parser

from .dataset_config import *
from .dataset import FusionDatasetFLIR, FusionDatasetM3FD

def create_dataset(name, root, splits=('train', 'val')):
    if isinstance(splits, str):
        splits = (splits,)
    name = name.lower()
    root = Path(root)
    
    # FLIR-Aligned Dataset
    if name == 'flir_aligned_full':
        dataset_cls = FusionDatasetFLIR
        datasets = OrderedDict()
        dataset_cfg = FlirAlignedFullCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('thermal', 'rgb')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )

    elif name == 'flir_aligned_day':
        dataset_cls = FusionDatasetFLIR
        datasets = OrderedDict()
        dataset_cfg = FlirAlignedDayCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('thermal', 'rgb')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )

    elif name == 'flir_aligned_night':
        dataset_cls = FusionDatasetFLIR
        datasets = OrderedDict()
        dataset_cfg = FlirAlignedNightCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('thermal', 'rgb')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )

    # M3FD Dataset
    elif name == 'm3fd_day':
        dataset_cls = FusionDatasetM3FD
        datasets = OrderedDict()
        dataset_cfg = M3fdDayCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('Ir', 'Vis')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )

    elif name == 'm3fd_night':
        dataset_cls = FusionDatasetM3FD
        datasets = OrderedDict()
        dataset_cfg = M3fdNightCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('Ir', 'Vis')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )

    elif name == 'm3fd_challenge':
        dataset_cls = FusionDatasetM3FD
        datasets = OrderedDict()
        dataset_cfg = M3fdChallengeCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('Ir', 'Vis')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )

    elif name == 'm3fd_full':
        dataset_cls = FusionDatasetM3FD
        datasets = OrderedDict()
        dataset_cfg = M3fdFullCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('Ir', 'Vis')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )

    elif name == 'm3fd_overcast':
        dataset_cls = FusionDatasetM3FD
        datasets = OrderedDict()
        dataset_cfg = M3fdOvercastCfg()
        for s in splits:
            if s not in dataset_cfg.splits:
                raise RuntimeError(f'{s} split not found in config')
            split_cfg = dataset_cfg.splits[s]
            ann_file = root / split_cfg['ann_filename']
            parser_cfg = CocoParserCfg(
                ann_filename=ann_file,
                has_labels=split_cfg['has_labels']
            )
            datasets[s] = dataset_cls(
                thermal_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('Ir', 'Vis')),
                parser=create_parser(dataset_cfg.parser, cfg=parser_cfg),
            )

    else:
        assert False, f'Unknown dataset parser ({name})'

    datasets = list(datasets.values())
    return datasets if len(datasets) > 1 else datasets[0]
