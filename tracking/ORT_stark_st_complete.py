import argparse
import torch
import _init_paths
from lib.models.stark.repvgg import repvgg_model_convert
from lib.models.stark import build_starkst
from lib.config.stark_st2.config import cfg, update_config_from_file
import torch.nn as nn
import torch.nn.functional as F
# for onnx conversion and inference
import torch.onnx
import numpy as np
import onnx
import onnxruntime
import time
import os
from lib.test.evaluation.environment import env_settings
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import Optional, List, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('script', type=str, default='stark_st', help='script name')
    parser.add_argument('config', type=str, default='baseline', help='yaml configure file name')
    args = parser.parse_args()
    return args


def get_data(bs=1, sz_x=256, hw_z=512, c=256):
    feat_vec_z = torch.randn(hw_z, bs, c, requires_grad=True)  # HWxBxC
    mask_vec_z = torch.rand(bs, hw_z, requires_grad=True) > 0.5  # BxHW
    pos_vec_z = torch.randn(hw_z, bs, c, requires_grad=True)  # HWxBxC
    return feat_vec_z, mask_vec_z, pos_vec_z

@torch.jit.script
def t1(tup: List[torch.Tensor]) -> torch.Tensor:
    return tup[0]
def box_xyxy_to_cxcywh(v:Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]):
    x, y, z = v
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)
class STARK(nn.Module):
    def __init__(self, transformer, box_head, cls_head):
        super(STARK, self).__init__()

        self.transformer = transformer
        self.box_head = box_head
        self.cls_head = cls_head
        hidden_dim = transformer.d_model
        num_queries = 1
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        print(self.query_embed)
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)



    def forward(self, feat: torch.Tensor, mask: torch.Tensor,
                pos: torch.Tensor, run_box_head:bool =False, run_cls_head:bool =False):
        output_embed, enc_mem = self.transformer(feat, mask, self.query_embed.weight, pos, return_encoder_output=True)

        out_dict = {}
        if run_cls_head:
            # forward the classification head
            out_dict.update({'pred_logits': self.cls_head(output_embed)[-1]})
        if run_box_head:
            # forward the box prediction head
            out_dict_box, outputs_coord = self.forward_box_head(output_embed, enc_mem)
            # merge results
            out_dict.update(out_dict_box)
            return out_dict, outputs_coord, output_embed
        else:
            return out_dict, None, output_embed

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        # adjust shape
        enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
        dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
        att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # run the corner head
        outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new}
        return out, outputs_coord_new

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    load_checkpoint = True
    save_name = "complete.onnx"
    # update cfg
    args = parse_args()
    yaml_fname = 'experiments/stark_st2/%s.yaml' % (args.config)
    print(yaml_fname)
    update_config_from_file(yaml_fname)
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    # build the stark model
    model = build_starkst(cfg)
    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        checkpoint_name = os.path.join(save_dir,
                                       "checkpoints/train/stark_st2/%s/STARKST_ep0004.pth.tar"
                                       % (args.config))
        model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=True)
    # transfer to test mode
    model = repvgg_model_convert(model)
    model.eval()
    """ rebuild the inference-time model """
    transformer = model.transformer
    box_head = model.box_head
    box_head.coord_x = box_head.coord_x.cpu()
    box_head.coord_y = box_head.coord_y.cpu()
    cls_head = model.cls_head
    '''
    box_head = model.box_head
    box_head.coord_x = box_head.coord_x.cpu()
    box_head.coord_y = box_head.coord_y.cpu()
    '''
    torch_model = STARK(transformer, box_head, cls_head)
    #print(torch_model)
    feat_vec_z, mask_vec_z, pos_vec_z = get_data()
    #torch_outs = torch_model(feat_vec_z, mask_vec_z, pos_vec_z, run_box_head=True, run_cls_head=True)
    torchscript_model = torch.jit.script(torch_model)
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized, "stark_st_transformer.pt")
    '''
    # get the network input
    bs = 1
    sz_x = cfg.TEST.SEARCH_SIZE
    hw_z = cfg.DATA.TEMPLATE.FEAT_SIZE ** 2
    c = cfg.MODEL.HIDDEN_DIM
    print(bs, sz_x, hw_z, c)
    img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z = get_data(bs=bs, sz_x=sz_x, hw_z=hw_z, c=c)
    torch_outs = torch_model(img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z)
    torch.onnx.export(torch_model,  # model being run
                      (img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z),  # model input (a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['img_x', 'mask_x', 'feat_vec_z', 'mask_vec_z', 'pos_vec_z'],  # model's input names
                      output_names=['outputs_coord'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    """########## inference with the pytorch model ##########"""
    # forward the template
    N = 1000
    torch_model = torch_model.cuda()
    torch_model.box_head.coord_x = torch_model.box_head.coord_x.cuda()
    torch_model.box_head.coord_y = torch_model.box_head.coord_y.cuda()

    '''