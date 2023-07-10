""" Fusion dataset for detection
"""
import torch.utils.data as data


from PIL import Image
from effdet.data.parsers import create_parser

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)




class FusionDatasetFLIR(data.Dataset):
    """ Fusion Dataset for Object Detection. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """
    def __init__(self, thermal_data_dir, rgb_data_dir, parser=None, parser_kwargs=None, transform=None):
        super(FusionDatasetFLIR, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.thermal_data_dir = thermal_data_dir
        self.rgb_data_dir = rgb_data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (thermal_image, rgb_image, annotations (target)).
        """
        # print(self._parser.img_infos)
        # index = 41
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        thermal_img_path = self.thermal_data_dir / img_info['file_name']
        thermal_img = Image.open(thermal_img_path).convert('RGB')
        rgb_img_path = self.rgb_data_dir / img_info['file_name'].replace('PreviewData', 'RGB')
        rgb_img = Image.open(rgb_img_path).convert('RGB')
        if self.transform is not None:
            thermal_img, rgb_img, target = self.transform(thermal_img, rgb_img, target)

        return thermal_img, rgb_img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t