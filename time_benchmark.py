import argparse
import torch
import tqdm
from models.models import Adaptive_Att_FusionNet, Att_FusionNet
from models.detector import DetBenchPredictImagePair, DetBenchPredict
import effdet 
from effdet import EfficientDet

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def run_effdet(args):
    # Benchmarking time: 
    # - Use fake data to easily run on multiple machines
    #   without worrying about copying the dataset around.
    # - No need to load weights since evaluation is not done, 
    #   just create the model.
    # - Use batch size of 1 to mimic robotic inference
    # - Warm up the network for 10 iterations in order to discard
    #   the first few iterations which are slower due to GPU initialization.
    device = torch.device('cuda:{}'.format(args.device))

    config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
    config.num_classes = args.num_classes
    model = EfficientDet(config)
    bench = DetBenchPredict(model)
    
    bench = bench.to(device)
    bench.eval()
    bench = torch.compile(bench, mode="reduce-overhead", backend="inductor")
    
    model_config = bench.config
    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (model_config.name, param_count))
    
    # Create dummy inputs
    thermal_input = torch.randn(1, 3, 768, 768).to(device)
    with torch.no_grad():
        for i in range(20):
            output = bench(thermal_input, img_info=None)
            
    total_time = 0
    with torch.no_grad():
        for i in tqdm.tqdm(range(100), total=100):
            _, t = timed(lambda: bench(thermal_input, img_info=None))
            total_time += t
            average_time = total_time / (i + 1)
    print('Average time (forward pass): {:4f} seconds'.format(average_time))
    print('Average freq (forward pass): {:4f} hz'.format(1 / average_time))    

def run(args):
    # Benchmarking time: 
    # - Use fake data to easily run on multiple machines
    #   without worrying about copying the dataset around.
    # - No need to load weights since evaluation is not done, 
    #   just create the model.
    # - Use batch size of 1 to mimic robotic inference
    # - Warm up the network for 10 iterations in order to discard
    #   the first few iterations which are slower due to GPU initialization.
    device = torch.device('cuda:{}'.format(args.device))
    if args.scene_mode == 'adaptive':
        model = Adaptive_Att_FusionNet(args)
    elif args.scene_mode == 'agnostic':
        model = Att_FusionNet(args)
    else:
        raise Exception('Mode is not valid')

    bench = DetBenchPredictImagePair(model)
    bench = bench.to(device)
    bench.eval()
    bench = torch.compile(bench, mode="reduce-overhead", backend="inductor")
    
    model_config = bench.config
    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (model_config.name, param_count))
    
    # Create dummy inputs
    thermal_input = torch.randn(1, 3, 768, 768).to(device)
    rgb_input = torch.randn(1, 3, 768, 768).to(device)

    # Warm up network
    with torch.no_grad():
        for i in range(20):
            output = bench(thermal_input, rgb_input, img_info=None, branch=args.branch)
            
    total_time = 0
    with torch.no_grad():
        for i in tqdm.tqdm(range(100), total=100):
            _, t = timed(lambda: bench(thermal_input, rgb_input, img_info=None, branch=args.branch))
            total_time += t
            average_time = total_time / (i + 1)
    print('Average time (forward pass): {:4f} seconds'.format(average_time))
    print('Average freq (forward pass): {:4f} hz'.format(1 / average_time))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
    parser.add_argument('--branch', default='fusion', type=str, metavar='BRANCH',
                        help='the inference branch ("thermal", "rgb", "fusion", or "single")')
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
    parser.add_argument('--scene-mode', type=str, default='adaptive', choices=['adaptive', 'agnostic'])
    parser.add_argument('--device', type=int, default=0, help="cuda device id")

    args = parser.parse_args()
    args.dataset = 'fake_benchmark_data'
    run(args)
    run_effdet(args)
