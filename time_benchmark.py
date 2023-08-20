#!/usr/bin/env python

import argparse
import time
import torch
import tqdm
from models.models import Adaptive_Att_FusionNet
from models.detector import DetBenchPredictImagePair

def run(args):
    # Benchmarking time: 
    # - Use fake data to easily run on multiple machines
    #   without worrying about copying the dataset around.
    # - No need to load weights since evaluation is not done, 
    #   just create the model.
    # - Use batch size of 1 to mimic robotic inference
    # - Warm up the network for 10 iterations in order to discard
    #   the first few iterations which are slower due to GPU initialization.

    model = Adaptive_Att_FusionNet(args)
    bench = DetBenchPredictImagePair(model)
    bench = bench.cuda()
    bench.eval()

    model_config = bench.config
    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (model_config.name, param_count))

    # Warm up network
    with torch.no_grad():
        for i in range(10):
            thermal_input = torch.randn(1, 3, 768, 768).cuda()
            rgb_input = torch.randn(1, 3, 768, 768).cuda()
            output = bench(thermal_input, rgb_input, img_info=None, branch=args.branch)
            
    total_time = 0
    with torch.no_grad():
        for i in tqdm.tqdm(range(100), total=100):
            thermal_input = torch.randn(1, 3, 768, 768).cuda()
            rgb_input = torch.randn(1, 3, 768, 768).cuda()
            start = time.time()
            output = bench(thermal_input, rgb_input, img_info=None, branch=args.branch)
            end = time.time()
            total_time += end - start
            average_time = total_time / (i + 1)
    print('Average time (forward pass): %.4f seconds' % average_time)    

if __name__ == '__main__':
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
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='Override num_classes in model config if set. For fine-tuning from pretrained.')
    parser.add_argument('--num-scenes', type=int, default=None, metavar='N',
                        help='Number of scene categories in dataset. For fusion module initialization.')
    parser.add_argument('--att_type', default='None', type=str, choices=['cbam','shuffle','eca'])
    parser.add_argument('--channels', default=128, type=int,
                            metavar='N', help='channels (default: 128)')
    parser.add_argument('--init-fusion-head-weights', type=str, default=None, choices=['thermal', 'rgb', None])
    parser.add_argument('--thermal-checkpoint-path', type=str, default=None)
    parser.add_argument('--rgb-checkpoint-path', type=str, default=None)
    args = parser.parse_args()
    run(args)

