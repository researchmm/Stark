"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from lib.utils.misc import is_main_process
from lib.models.stark import resnet as resnet_module
from lib.models.stark.repvgg import get_RepVGG_func_by_name
import os
from .swin_transformer import build_swint


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


class Transformer_Backbone(nn.Module):
    """Transformer Backbone"""

    def __init__(self, img_sz, model_name, train_backbone):
        super().__init__()
        if model_name == "vit_deit_base_distilled_patch16_384":
            ckpt_name = "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth"
            model = deit(img_sz, pretrained=is_main_process(), model_name=model_name, ckpt_name=ckpt_name)
            self.num_channels = model.num_features
        elif model_name == "swin_base_patch4_window12_384_S16":
            model = build_swint(model_name)
            self.num_channels = model.num_features[-1]
        else:
            raise ValueError("Unsupported model name")

        if not train_backbone:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(False)
        self.body = model

    def forward(self, x: torch.Tensor):
        return self.body(x)


class Backbone(nn.Module):
    def __init__(self, name: str, dilation: bool, freeze_bn: bool, last_stage_block=14):
        super().__init__()
        if "resnet" in name:
            norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
            # here is different from the original DETR because we use feature from block3
            self.body = getattr(resnet_module, name)(
                replace_stride_with_dilation=[False, dilation, False],
                pretrained=is_main_process(), norm_layer=norm_layer, last_layer='layer3')
            self.num_channels = 256 if name in ('resnet18', 'resnet34') else 1024
        elif "RepVGG" in name:
            print("#" * 10 + "  Warning: Dilation is not valid in current code  " + "#" * 10)
            repvgg_func = get_RepVGG_func_by_name(name)
            self.body = repvgg_func(deploy=False, last_layer="stage3", freeze_bn=freeze_bn,
                                    last_stage_block=last_stage_block)
            self.num_channels = 192  # 256x0.75=192
        elif "LightTrack" in name:
            print("#" * 10 + "  Warning: Dilation is not valid in current code  " + "#" * 10)
            path_backbone = [[0], [4, 5], [0, 2, 5, 1], [4, 0, 4, 4], [5, 2, 1, 0]]
            ops = (3, 2)
            self.body = build_subnet(path_backbone, ops)
            if is_main_process():
                print(self.body)
            self.num_channels = 96
        else:
            raise ValueError("Unsupported net type")

    def forward(self, x: torch.Tensor):
        return self.body(x)


def build_backbone_x_cnn(cfg, phase='train'):
    """Without positional embedding, standard tensor input"""
    train_backbone = cfg.TRAIN.BACKBONE_MULTIPLIER > 0
    backbone = Backbone(cfg.MODEL.BACKBONE.TYPE, cfg.MODEL.BACKBONE.DILATION, cfg.TRAIN.FREEZE_BACKBONE_BN,
                        cfg.MODEL.BACKBONE.LAST_STAGE_BLOCK)

    if phase is 'train':
        """load pretrained backbone weights"""
        ckpt_path = None
        if hasattr(cfg, "ckpt_dir"):
            if cfg.MODEL.BACKBONE.TYPE == "RepVGG-A0":
                filename = "RepVGG-A0-train.pth"
            elif cfg.MODEL.BACKBONE.TYPE == "LightTrack":
                filename = "LightTrackM.pth"
            else:
                raise ValueError("The checkpoint file for backbone type %s is not found" % cfg.MODEL.BACKBONE.TYPE)
            ckpt_path = os.path.join(cfg.ckpt_dir, filename)
        if ckpt_path is not None:
            print("Loading pretrained backbone weights from %s" % ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            if cfg.MODEL.BACKBONE.TYPE == "LightTrack":
                ckpt, ckpt_new = ckpt["state_dict"], {}
                for k, v in ckpt.items():
                    if k.startswith("features."):
                        k_new = k.replace("features.", "")
                        ckpt_new[k_new] = v
                ckpt = ckpt_new
            missing_keys, unexpected_keys = backbone.body.load_state_dict(ckpt, strict=False)
            if is_main_process():
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)

        """freeze some layers"""
        if cfg.MODEL.BACKBONE.TYPE != "LightTrack":
            trained_layers = cfg.TRAIN.BACKBONE_TRAINED_LAYERS
            # defrost parameters of layers in trained_layers
            for name, parameter in backbone.body.named_parameters():
                parameter.requires_grad_(False)
                if train_backbone:
                    for trained_name in trained_layers:  # such as 'layer2' in layer2.conv1.weight
                        if trained_name in name:
                            parameter.requires_grad_(True)
                            break
    return backbone


def build_backbone_x_swin(cfg, phase='train'):
    """Without positional embedding, standard tensor input"""
    train_backbone = cfg.TRAIN.BACKBONE_MULTIPLIER > 0
    if cfg.MODEL.BACKBONE.TYPE == "swin_base_patch4_window12_384_S16":
        backbone = Transformer_Backbone(cfg.DATA.SEARCH.SIZE, cfg.MODEL.BACKBONE.TYPE, train_backbone)
    else:
        raise ValueError("Unsupported model_name")

    if phase is 'train':
        """load pretrained backbone weights"""
        ckpt_path = None
        if hasattr(cfg, "ckpt_dir"):
            if cfg.MODEL.BACKBONE.TYPE == "swin_base_patch4_window12_384_S16":
                filename = "swin_base_patch4_window12_384_22k.pth"
            else:
                raise ValueError("The checkpoint file for backbone type %s is not found" % cfg.MODEL.BACKBONE.TYPE)
            ckpt_path = os.path.join(cfg.ckpt_dir, filename)
        if ckpt_path is not None:
            print("Loading pretrained backbone weights from %s" % ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            missing_keys, unexpected_keys = backbone.body.load_state_dict(ckpt["model"], strict=False)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            del ckpt
            torch.cuda.empty_cache()
    print("For swin, we don't freeze any layers in the backbone ")
    return backbone


def build_backbone_x(cfg, phase='train'):
    if 'swin' in cfg.MODEL.BACKBONE.TYPE:
        return build_backbone_x_swin(cfg, phase=phase)
    else:
        return build_backbone_x_cnn(cfg, phase=phase)
