
from .transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def resolve_input_config(args, model_config=None, model=None):
    if not isinstance(args, dict):
        args = vars(args)
    input_config = {}
    if not model_config and model is not None and hasattr(model, 'config'):
        model_config = model.config

    # Resolve input/image size
    in_chans = 3
    # if 'chans' in args and args['chans'] is not None:
    #     in_chans = args['chans']

    input_size = (in_chans, 512, 512)
    # if 'input_size' in args and args['input_size'] is not None:
    #     assert isinstance(args['input_size'], (tuple, list))
    #     assert len(args['input_size']) == 3
    #     input_size = tuple(args['input_size'])
    #     in_chans = input_size[0]  # input_size overrides in_chans
    # elif 'img_size' in args and args['img_size'] is not None:
    #     assert isinstance(args['img_size'], int)
    #     input_size = (in_chans, args['img_size'], args['img_size'])
    if 'input_size' in model_config:
        input_size = tuple(model_config['input_size'])
    elif 'image_size' in model_config:
        input_size = (in_chans,) + tuple(model_config['image_size'])
    assert isinstance(input_size, tuple) and len(input_size) == 3
    input_config['input_size'] = input_size

    # resolve interpolation method
    input_config['interpolation'] = 'bicubic'
    if 'interpolation' in args and args['interpolation']:
        input_config['interpolation'] = args['interpolation']
    elif 'interpolation' in model_config:
        input_config['interpolation'] = model_config['interpolation']

    # resolve dataset + model mean for normalization
    input_config['rgb_mean'] = IMAGENET_DEFAULT_MEAN
    if 'rgb_mean' in args and args['rgb_mean'] is not None:
        rgb_mean = tuple(args['rgb_mean'])
        if len(rgb_mean) == 1:
            rgb_mean = tuple(list(rgb_mean) * in_chans)
        else:
            assert len(rgb_mean) == in_chans
        input_config['rgb_mean'] = rgb_mean
    elif 'mean' in model_config:
        input_config['rgb_mean'] = model_config['mean']

    # resolve dataset + model std deviation for normalization
    input_config['rgb_std'] = IMAGENET_DEFAULT_STD
    if 'rgb_std' in args and args['rgb_std'] is not None:
        rgb_std = tuple(args['rgb_std'])
        if len(rgb_std) == 1:
            rgb_std = tuple(list(rgb_std) * in_chans)
        else:
            assert len(rgb_std) == in_chans
        input_config['rgb_std'] = rgb_std
    elif 'std' in model_config:
        input_config['rgb_std'] = model_config['std']

    # resolve dataset + model mean for normalization
    input_config['thermal_mean'] = IMAGENET_DEFAULT_MEAN
    if 'thermal_mean' in args and args['thermal_mean'] is not None:
        thermal_mean = tuple(args['thermal_mean'])
        if len(thermal_mean) == 1:
            thermal_mean = tuple(list(thermal_mean) * in_chans)
        else:
            assert len(thermal_mean) == in_chans
        input_config['thermal_mean'] = thermal_mean
    elif 'mean' in model_config:
        input_config['thermal_mean'] = model_config['mean']

    # resolve dataset + model std deviation for normalization
    input_config['thermal_std'] = IMAGENET_DEFAULT_STD
    if 'thermal_std' in args and args['thermal_std'] is not None:
        thermal_std = tuple(args['thermal_std'])
        if len(thermal_std) == 1:
            thermal_std = tuple(list(thermal_std) * in_chans)
        else:
            assert len(thermal_std) == in_chans
        input_config['thermal_std'] = thermal_std
    elif 'std' in model_config:
        input_config['thermal_std'] = model_config['std']

    # resolve letterbox fill color
    input_config['fill_color'] = 'mean'
    if 'fill_color' in args and args['fill_color'] is not None:
        input_config['fill_color'] = args['fill_color']
    elif 'fill_color' in model_config:
        input_config['fill_color'] = model_config['fill_color']

    return input_config