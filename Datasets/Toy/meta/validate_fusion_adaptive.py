#!/usr/bin/env python

import argparse
import time
import torch
import torch.nn.parallel
from contextlib import suppress

from effdet import create_model, create_evaluator
from timm.utils import AverageMeter, setup_default_logging
from timm.models import load_checkpoint
from timm.models.layers import set_layer_config

from models.models import Att_FusionNet, Adaptive_Att_FusionNet
from models.detector import DetBenchPredictImagePair
from data import create_dataset, create_loader, resolve_input_config
from utils.evaluator import create_evaluator
from utils.utils import load_checkpoint_selective
from utils.utils import visualize_detections

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--branch', default='fusion', type=str, metavar='BRANCH',
                    help='the inference branch ("thermal", "rgb", "fusion", or "single")')
parser.add_argument('root', metavar='DIR',
                    help='path to dataset root')
parser.add_argument('--dataset', default='flir_aligned', type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='test',
                    help='test split')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('--num-scenes', type=int, default=None, metavar='N',
                    help='Number of scene categories in dataset. For fusion module initialization.')
parser.add_argument('--att_type', default='None', type=str, choices=['cbam','shuffle','eca'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--rgb_mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of RGB dataset')
parser.add_argument('--rgb_std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of RGB dataset')
parser.add_argument('--thermal_mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of Thermal dataset')
parser.add_argument('--thermal_std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of Thermal dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint-cls', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint-scenes', type=str, nargs='+', default=None, metavar='PATH',
                    help='path to scene-specific checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--channels', default=128, type=int,
                        metavar='N', help='channels (default: 128)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')
parser.add_argument('--init-fusion-head-weights', type=str, default=None, choices=['thermal', 'rgb', None])
parser.add_argument('--thermal-checkpoint-path', type=str, default=None)
parser.add_argument('--rgb-checkpoint-path', type=str, default=None)
parser.add_argument('--classwise', dest='classwise', action='store_true',
                    help='use Pascal evaluator for classwise metrics')
parser.add_argument('--wandb', action='store_true',
                    help='use wandb for logging and visualization')


def validate(args):
    setup_default_logging()

    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    args.pretrained = args.pretrained or not args.checkpoint  # might as well try to validate something
    args.prefetcher = not args.no_prefetcher

    # create model
    if args.branch == 'single':
        with set_layer_config(scriptable=args.torchscript):
            extra_args = {}
            if args.img_size is not None:
                extra_args = dict(image_size=(args.img_size, args.img_size))
            bench = create_model(
                args.model,
                bench_task='predict',
                num_classes=args.num_classes,
                pretrained=args.pretrained,
                redundant_bias=args.redundant_bias,
                soft_nms=args.soft_nms,
                checkpoint_path=args.checkpoint,
                checkpoint_ema=args.use_ema,
                **extra_args,
            )
    else:
        model = Adaptive_Att_FusionNet(args)
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint, use_ema=args.use_ema, strict=False)
        if args.checkpoint_cls:
            load_checkpoint_selective(model, args.checkpoint_cls)
        if args.num_scenes:
            for scene in range(args.num_scenes):
                load_checkpoint_selective(model, args.checkpoint_scenes[scene], scene)
        bench = DetBenchPredictImagePair(model)
    model_config = bench.config

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (model_config.name, param_count))

    bench = bench.cuda()

    amp_autocast = suppress
    if args.apex_amp:
        bench = amp.initialize(bench, opt_level='O1')
        print('Using NVIDIA APEX AMP. Validating in mixed precision.')
    elif args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        print('Using native Torch AMP. Validating in mixed precision.')
    else:
        print('AMP not enabled. Validating in float32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

    dataset = create_dataset(args.dataset, args.root, args.split)
    input_config = resolve_input_config(args, model_config)
    loader = create_loader(
        dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        rgb_mean=input_config['rgb_mean'],
        rgb_std=input_config['rgb_std'],
        thermal_mean=input_config['thermal_mean'],
        thermal_std=input_config['thermal_std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    
    evaluator = create_evaluator(args.dataset, dataset, distributed=False, pred_yxyx=False)


    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1

        # logging
    if args.wandb:
        import wandb
        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
          project='wacv2024',
          config=config
        )

    with torch.no_grad():
        for i, (thermal_input, rgb_input, target) in enumerate(loader):
            with amp_autocast():
                if args.branch == 'single':
                    output = bench(thermal_input, img_info=target)
                else:
                    output = bench(thermal_input, rgb_input, img_info=target, branch=args.branch)
            evaluator.add_predictions(output, target)
            if args.wandb:
                visualize_detections(dataset, output, target, wandb, args, 'test')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0 or i == last_idx:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=thermal_input.size(0) / batch_time.avg)
                )

    mean_ap = 0.
    if dataset.parser.has_labels:
        mean_ap = evaluator.evaluate(output_result_file=args.results)
    else:
        evaluator.save(args.results)

    return mean_ap


def main():
    args = parser.parse_args()

    print("Dataset: "+args.dataset)
    if args.checkpoint == '':

        print("Branch: "+args.branch)

    else:
        print("Checkpoint: "+args.checkpoint)
        print("Att Type: "+args.att_type)

    
    mean_ap = validate(args)
    print("*"*50)
    print("Mean Average Precision Obtained is : "+str(mean_ap))
    print("*"*50)


if __name__ == '__main__':
    main()

