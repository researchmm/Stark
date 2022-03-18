import argparse
import torch
import _init_paths
from lib.models.stark.repvgg import repvgg_model_convert
from lib.models.stark import build_starkst
from lib.config.stark_st2.config import cfg, update_config_from_file
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np
import onnx
import onnxruntime
import time
import os
from lib.test.evaluation.environment import env_settings
from lib.utils.misc import *
from torch.utils.mobile_optimizer import optimize_for_mobile

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    #parser.add_argument('--script', type=str, default='stark_lightning_X_trt', help='script name')
    #parser.add_argument('--config', type=str, default='baseline_rephead_4_lite_search5', help='yaml configure file name')
    parser.add_argument('--script', type=str, default='stark_st2', help='script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    args = parser.parse_args()
    return args


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz, requires_grad=True)
    mask = torch.rand(bs, sz, sz, requires_grad=True) > 0.5
    return img_patch, mask


class Backbone_Bottleneck(nn.Module):
    def __init__(self, backbone, bottleneck):
        super(Backbone_Bottleneck, self).__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck

    def forward(self, input: NestedTensor):
        assert isinstance(input, NestedTensor)
        output_back, pos = self.backbone(input)
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    load_checkpoint = True
    save_name = "backbone_stark_st.onnx"
    """update cfg"""
    args = parse_args()
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    print(yaml_fname)
    update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    # build the stark model
    model = build_starkst(cfg)
    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        checkpoint_name = os.path.join(save_dir,
                                       "checkpoints/train/%s/%s/STARKST_ep0004.pth.tar"
                                       % (args.script, args.config))
        model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=True)
    # transfer to test mode
    model = repvgg_model_convert(model)
    model.eval()
    """ rebuild the inference-time model """
    backbone = model.backbone
    bottleneck = model.bottleneck
    torch_model = Backbone_Bottleneck(backbone, bottleneck)
    print(torch_model)
    # get the template
    img_z, mask_z = get_data(bs, z_sz)
    # forward the template
    torch_outs = torch_model(NestedTensor(img_z, mask_z))
    '''
    torch.onnx.export(torch_model,  # model being run
                      NestedTensor(img_z, mask_z),  # model input (or a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['img_z', 'mask_z'],  # the model's input names
                      output_names=['feat', 'mask', 'pos'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    '''

    torchscript_model = torch.jit.script(torch_model)
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized, "stark_st_backbone.pt")
