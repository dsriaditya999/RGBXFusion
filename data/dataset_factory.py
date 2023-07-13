""" Dataset factory
"""
from collections import OrderedDict
from pathlib import Path

from effdet.data.parsers import *
from effdet.data.parsers import create_parser
from .parsers import create_parser as create_parser_stf

from .dataset_config import *
from .dataset import FusionDatasetFLIR, FusionDatasetM3FD, XBitFusionDatsetSTF

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


# STF Dataset

    elif name == 'stf_full': 
        dataset_cls = XBitFusionDatsetSTF
        datasets = OrderedDict()
        dataset_cfg = StfFullCfg()
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
                gated_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('gated_full_acc_rect_aligned', 'cam_stereo_left_rect_aligned')),
                parser=create_parser_stf(dataset_cfg.parser, cfg=parser_cfg),
                mode=s, 
                rgb_bits=12,
                gated_bits=10,
                rgb_mean = [0.26694615, 0.26693442, 0.26698295], 
                rgb_std = [0.12035122, 0.12039929, 0.12037755],
                gated_mean = [0.20945697, 0.20945697, 0.20945697], 
                gated_std = [0.15437697, 0.15437697, 0.15437697]
            )


    elif name == 'stf_clear': 
        dataset_cls = XBitFusionDatsetSTF
        datasets = OrderedDict()
        dataset_cfg = StfClearCfg()
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
                gated_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('gated_full_acc_rect_aligned', 'cam_stereo_left_rect_aligned')),
                parser=create_parser_stf(dataset_cfg.parser, cfg=parser_cfg),
                mode=s, 
                rgb_bits=12,
                gated_bits=10,
                rgb_mean = [0.26694615, 0.26693442, 0.26698295], 
                rgb_std = [0.12035122, 0.12039929, 0.12037755],
                # rgb_mean= [0.519, 0.519, 0.519],
                # rgb_std=[0.225, 0.225, 0.225],
                gated_mean = [0.20945697, 0.20945697, 0.20945697], 
                gated_std = [0.15437697, 0.15437697, 0.15437697]
            )

    elif name == 'stf_clear_day': 
        dataset_cls = XBitFusionDatsetSTF
        datasets = OrderedDict()
        dataset_cfg = StfClearDayCfg()
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
                gated_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('gated_full_acc_rect_aligned', 'cam_stereo_left_rect_aligned')),
                parser=create_parser_stf(dataset_cfg.parser, cfg=parser_cfg),
                mode=s, 
                rgb_bits=12,
                gated_bits=10,
                rgb_mean = [0.26694615, 0.26693442, 0.26698295], 
                rgb_std = [0.12035122, 0.12039929, 0.12037755],
                # rgb_mean= [0.519, 0.519, 0.519],
                # rgb_std=[0.225, 0.225, 0.225],
                gated_mean = [0.20945697, 0.20945697, 0.20945697], 
                gated_std = [0.15437697, 0.15437697, 0.15437697]
            )

    elif name == 'stf_clear_night': 
        dataset_cls = XBitFusionDatsetSTF
        datasets = OrderedDict()
        dataset_cfg = StfClearNightCfg()
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
                gated_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('gated_full_acc_rect_aligned', 'cam_stereo_left_rect_aligned')),
                parser=create_parser_stf(dataset_cfg.parser, cfg=parser_cfg),
                mode=s, 
                rgb_bits=12,
                gated_bits=10,
                rgb_mean = [0.26694615, 0.26693442, 0.26698295], 
                rgb_std = [0.12035122, 0.12039929, 0.12037755],
                # rgb_mean= [0.519, 0.519, 0.519],
                # rgb_std=[0.225, 0.225, 0.225],
                gated_mean = [0.20945697, 0.20945697, 0.20945697], 
                gated_std = [0.15437697, 0.15437697, 0.15437697]
            )

    elif name == 'stf_fog_day': 
        dataset_cls = XBitFusionDatsetSTF
        datasets = OrderedDict()
        dataset_cfg = StfFogDayCfg()
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
                gated_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('gated_full_acc_rect_aligned', 'cam_stereo_left_rect_aligned')),
                parser=create_parser_stf(dataset_cfg.parser, cfg=parser_cfg),
                mode=s, 
                rgb_bits=12,
                gated_bits=10,
                rgb_mean = [0.26694615, 0.26693442, 0.26698295], 
                rgb_std = [0.12035122, 0.12039929, 0.12037755],
                gated_mean = [0.20945697, 0.20945697, 0.20945697], 
                gated_std = [0.15437697, 0.15437697, 0.15437697]
            )

    elif name == 'stf_fog_night': 
        dataset_cls = XBitFusionDatsetSTF
        datasets = OrderedDict()
        dataset_cfg = StfFogNightCfg()
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
                gated_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('gated_full_acc_rect_aligned', 'cam_stereo_left_rect_aligned')),
                parser=create_parser_stf(dataset_cfg.parser, cfg=parser_cfg),
                mode=s, 
                rgb_bits=12,
                gated_bits=10,
                rgb_mean = [0.26694615, 0.26693442, 0.26698295], 
                rgb_std = [0.12035122, 0.12039929, 0.12037755],
                gated_mean = [0.20945697, 0.20945697, 0.20945697], 
                gated_std = [0.15437697, 0.15437697, 0.15437697]
            )

    elif name == 'stf_snow_day': 
        dataset_cls = XBitFusionDatsetSTF
        datasets = OrderedDict()
        dataset_cfg = StfSnowDayCfg()
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
                gated_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('gated_full_acc_rect_aligned', 'cam_stereo_left_rect_aligned')),
                parser=create_parser_stf(dataset_cfg.parser, cfg=parser_cfg),
                mode=s, 
                rgb_bits=12,
                gated_bits=10,
                rgb_mean = [0.26694615, 0.26693442, 0.26698295], 
                rgb_std = [0.12035122, 0.12039929, 0.12037755],
                gated_mean = [0.20945697, 0.20945697, 0.20945697], 
                gated_std = [0.15437697, 0.15437697, 0.15437697]
            )

    elif name == 'stf_snow_night': 
        dataset_cls = XBitFusionDatsetSTF
        datasets = OrderedDict()
        dataset_cfg = StfSnowNightCfg()
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
                gated_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('gated_full_acc_rect_aligned', 'cam_stereo_left_rect_aligned')),
                parser=create_parser_stf(dataset_cfg.parser, cfg=parser_cfg),
                mode=s, 
                rgb_bits=12,
                gated_bits=10,
                rgb_mean = [0.26694615, 0.26693442, 0.26698295], 
                rgb_std = [0.12035122, 0.12039929, 0.12037755],
                gated_mean = [0.20945697, 0.20945697, 0.20945697], 
                gated_std = [0.15437697, 0.15437697, 0.15437697]
            )

    elif name == 'stf_rain': 
        dataset_cls = XBitFusionDatsetSTF
        datasets = OrderedDict()
        dataset_cfg = StfRainCfg()
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
                gated_data_dir=root / Path(split_cfg['img_dir']),
                rgb_data_dir=root / Path(split_cfg['img_dir'].replace('gated_full_acc_rect_aligned', 'cam_stereo_left_rect_aligned')),
                parser=create_parser_stf(dataset_cfg.parser, cfg=parser_cfg),
                mode=s, 
                rgb_bits=12,
                gated_bits=10,
                rgb_mean = [0.26694615, 0.26693442, 0.26698295], 
                rgb_std = [0.12035122, 0.12039929, 0.12037755],
                gated_mean = [0.20945697, 0.20945697, 0.20945697], 
                gated_std = [0.15437697, 0.15437697, 0.15437697]
            )

    else:
        assert False, f'Unknown dataset parser ({name})'

    datasets = list(datasets.values())
    return datasets if len(datasets) > 1 else datasets[0]
