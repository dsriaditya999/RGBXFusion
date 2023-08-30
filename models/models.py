import torch
import torch.nn as nn

import effdet
from effdet import EfficientDet


from models.fusion_modules import CBAMLayer, attention_block, shuffle_attention_block

##################################### Attention Fusion Net ###############################################
class Att_FusionNet(nn.Module):

    def __init__(self, args):
        super(Att_FusionNet, self).__init__()

        self.config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
        self.config.num_classes = args.num_classes
        if "stf" in args.dataset:
            self.config.image_size = (1280, 1280)

        thermal_det = EfficientDet(self.config)
        rgb_det = EfficientDet(self.config)

        if args.thermal_checkpoint_path:
            effdet.helpers.load_checkpoint(thermal_det, args.thermal_checkpoint_path)
            print('Loading Thermal from {}'.format(args.thermal_checkpoint_path))
        else:
            print('Thermal checkpoint path not provided.')
        
        if args.rgb_checkpoint_path:
            effdet.helpers.load_checkpoint(rgb_det, args.rgb_checkpoint_path)
            print('Loading RGB from {}'.format(args.rgb_checkpoint_path))
        else:
            if 'flir' in args.dataset:
                effdet.helpers.load_pretrained(rgb_det, self.config.url)
                print('Loading RGB from {}'.format(self.config.url))
            print('RGB checkpoint path not provided.')
            

            
        
        self.thermal_backbone = thermal_det.backbone
        self.thermal_fpn = thermal_det.fpn
        self.thermal_class_net = thermal_det.class_net
        self.thermal_box_net = thermal_det.box_net

        self.rgb_backbone = rgb_det.backbone
        self.rgb_fpn = rgb_det.fpn
        self.rgb_class_net = rgb_det.class_net
        self.rgb_box_net = rgb_det.box_net

        fusion_det = EfficientDet(self.config)
        
        if args.init_fusion_head_weights == 'thermal':
            effdet.helpers.load_checkpoint(fusion_det, args.thermal_checkpoint_path) # This is optional
            print("Loading fusion head from thermal checkpoint.")
        elif args.init_fusion_head_weights == 'rgb':
            effdet.helpers.load_checkpoint(fusion_det, args.rgb_checkpoint_path)
            print("Loading fusion head from rgb checkpoint.")
        else:
            print('Fusion head random init.')
        

        self.fusion_class_net = fusion_det.class_net
        self.fusion_box_net = fusion_det.box_net

        if args.branch == 'fusion':
            self.attention_type = args.att_type
            print("Using {} attention.".format(self.attention_type))
            in_chs = args.channels
            for level in range(self.config.num_levels):
                if self.attention_type=="cbam":
                    self.add_module("fusion_"+self.attention_type+str(level), CBAMLayer(2*in_chs))
                elif self.attention_type=="eca":
                    self.add_module("fusion_"+self.attention_type+str(level), attention_block(2*in_chs))
                elif self.attention_type=="shuffle":
                    self.add_module("fusion_"+self.attention_type+str(level), shuffle_attention_block(2*in_chs))
                else:
                    raise ValueError('Attention type not supported.')

    def forward(self, data_pair, branch='fusion'):
        thermal_x, rgb_x = data_pair[0], data_pair[1]

        class_net = getattr(self, f'{branch}_class_net')
        box_net = getattr(self, f'{branch}_box_net')
        
        x = None
        if branch =='fusion':
            thermal_x = self.thermal_backbone(thermal_x)
            rgb_x = self.rgb_backbone(rgb_x)

            thermal_x = self.thermal_fpn(thermal_x)
            rgb_x = self.rgb_fpn(rgb_x)

            out = []
            for i, (tx, vx) in enumerate(zip(thermal_x, rgb_x)):
                x = torch.cat((tx, vx), dim=1)
                attention = getattr(self, "fusion_"+self.attention_type+str(i))
                out.append(attention(x))
        else:
            fpn = getattr(self, f'{branch}_fpn')
            backbone = getattr(self, f'{branch}_backbone')
            if branch =='thermal':
                x = thermal_x
            elif branch =='rgb':
                x = rgb_x
            feats = backbone(x)
            out = fpn(feats)
        
        
        x_class = class_net(out)
        x_box = box_net(out)

        return x_class, x_box




##################################### Adaptive Fusion Net ###############################################
class Classifier(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(208, n_classes)

    def forward(self, x):
        x = self.l1(x)
        return x

class Adaptive_Att_FusionNet(Att_FusionNet):

    def __init__(self, args):
        Att_FusionNet.__init__(self, args)

        self.num_scenes = args.num_scenes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(n_classes=self.num_scenes, dropout=0.5)

        if args.branch == 'fusion':
            in_chs = args.channels
            del self.fusion_cbam0
            del self.fusion_cbam1
            del self.fusion_cbam2
            del self.fusion_cbam3
            del self.fusion_cbam4
            for scene in range(self.num_scenes):
                for level in range(self.config.num_levels):
                    if self.attention_type=="cbam":
                        self.add_module("fusion"+str(scene)+"_"+self.attention_type+str(level), CBAMLayer(2*in_chs))
                    elif self.attention_type=="eca":
                        self.add_module("fusion"+str(scene)+"_"+self.attention_type+str(level), attention_block(2*in_chs))
                    elif self.attention_type=="shuffle":
                        self.add_module("fusion"+str(scene)+"_"+self.attention_type+str(level), shuffle_attention_block(2*in_chs))
                    else:
                        raise ValueError('Attention type not supported.')

    def forward(self, data_pair, branch='fusion'):
        thermal_x, rgb_x = data_pair[0], data_pair[1]

        class_net = getattr(self, f'{branch}_class_net')
        box_net = getattr(self, f'{branch}_box_net')
        
        x = None
        if branch =='fusion':
            thermal_x = self.thermal_backbone(thermal_x)
            rgb_x = self.rgb_backbone(rgb_x)

            feat = self.avgpool(rgb_x[len(rgb_x)-1])
            feat = feat.view(feat.size(0), -1)
            image_class_out = self.classifier(feat)
            image_class_out = torch.argmax(image_class_out, dim=1).cpu().numpy()[0]

            thermal_x = self.thermal_fpn(thermal_x)
            rgb_x = self.rgb_fpn(rgb_x)

            out = []
            for i, (tx, vx) in enumerate(zip(thermal_x, rgb_x)):
                x = torch.cat((tx, vx), dim=1)
                attention = getattr(self, "fusion"+str(image_class_out)+"_"+self.attention_type+str(i))
                out.append(attention(x))
        else:
            fpn = getattr(self, f'{branch}_fpn')
            backbone = getattr(self, f'{branch}_backbone')
            if branch =='thermal':
                x = thermal_x
            elif branch =='rgb':
                x = rgb_x
            feats = backbone(x)
            out = fpn(feats)
        
        
        x_class = class_net(out)
        x_box = box_net(out)

        return x_class, x_box


##################################### Scene Classifier ###############################################
class EfficientDetwithCls(EfficientDet):

    def __init__(self, config, pretrained_backbone=True, alternate_init=False):
        EfficientDet.__init__(self, config, pretrained_backbone, alternate_init)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(n_classes=config.num_scenes, dropout=0.5)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.fpn.parameters():
            param.requires_grad = False
        for param in self.class_net.parameters():
            param.requires_grad = False
        for param in self.box_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        feat = self.avgpool(x[len(x)-1])
        feat = feat.view(feat.size(0), -1)
        image_class_out = self.classifier(feat)

        x = self.fpn(x)
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box, image_class_out