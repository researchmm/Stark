import argparse
import torch
import _init_paths
from lib.models.stark.repvgg import repvgg_model_convert
from lib.models.stark import build_stark_lightning_x_trt
from lib.config.stark_lightning_X_trt.config import cfg, update_config_from_file
from lib.utils.box_ops import box_xyxy_to_cxcywh
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


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--script', type=str, default='stark_lightning_X_trt', help='script name')
    parser.add_argument('--config', type=str, default='baseline_rephead_4_lite_search5', help='yaml configure file name')
    args = parser.parse_args()
    return args


def get_data(bs=1, sz_x=256, hw_z=64, c=256):
    img_x = torch.randn(bs, 3, sz_x, sz_x, requires_grad=True)
    mask_x = torch.rand(bs, sz_x, sz_x, requires_grad=True) > 0.5
    feat_vec_z = torch.randn(hw_z, bs, c, requires_grad=True)  # HWxBxC
    mask_vec_z = torch.rand(bs, hw_z, requires_grad=True) > 0.5  # BxHW
    pos_vec_z = torch.randn(hw_z, bs, c, requires_grad=True)  # HWxBxC
    return img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z


class STARK(nn.Module):
    def __init__(self, backbone, bottleneck, position_embed, transformer, box_head):
        super(STARK, self).__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck
        self.position_embed = position_embed
        self.transformer = transformer
        self.box_head = box_head
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, img: torch.Tensor, mask: torch.Tensor,
                feat_vec_z: torch.Tensor, mask_vec_z: torch.Tensor, pos_vec_z: torch.Tensor):
        # run the backbone
        feat = self.bottleneck(self.backbone(img))  # BxCxHxW
        mask_down = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.position_embed(bs=1)  # 1 is the batch-size. output size is BxCxHxW
        # adjust shape
        feat_vec_x = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_vec_x = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec_x = mask_down.flatten(1)  # BxHW
        # concat with the template-related results
        feat_vec = torch.cat([feat_vec_z, feat_vec_x], dim=0)
        mask_vec = torch.cat([mask_vec_z, mask_vec_x], dim=1)
        pos_vec = torch.cat([pos_vec_z, pos_vec_x], dim=0)
        # get q, k, v
        q = feat_vec_x + pos_vec_x
        k = feat_vec + pos_vec
        v = feat_vec
        key_padding_mask = mask_vec
        # run the transformer encoder
        memory = self.transformer(q, k, v, key_padding_mask=key_padding_mask)
        fx = memory[-self.feat_len_s:].permute(1, 2, 0).contiguous()  # (B, C, H_x*W_x)
        fx_t = fx.view(*fx.shape[:2], self.feat_sz_s, self.feat_sz_s).contiguous()  # fx tensor 4D (B, C, H_x, W_x)
        # run the corner head
        outputs_coord = box_xyxy_to_cxcywh(self.box_head(fx_t))
        return outputs_coord


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    load_checkpoint = True
    save_name = "complete.onnx"
    # update cfg
    args = parse_args()
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(yaml_fname)
    # build the stark model
    model = build_stark_lightning_x_trt(cfg, phase='test')
    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        checkpoint_name = os.path.join(save_dir,
                                       "checkpoints/train/%s/%s/STARKLightningXtrt_ep0500.pth.tar"
                                       % (args.script, args.config))
        model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=True)
    # transfer to test mode
    model = repvgg_model_convert(model)
    model.eval()
    """ rebuild the inference-time model """
    backbone = model.backbone
    bottleneck = model.bottleneck
    position_embed = model.pos_emb_x
    transformer = model.transformer
    box_head = model.box_head
    box_head.coord_x = box_head.coord_x.cpu()
    box_head.coord_y = box_head.coord_y.cpu()
    torch_model = STARK(backbone, bottleneck, position_embed, transformer, box_head)
    print(torch_model)
    torch.save(torch_model.state_dict(), "complete.pth")
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

    """########## inference with the onnx model ##########"""
    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)
    print("creating session...")
    ort_session = onnxruntime.InferenceSession(save_name)
    # ort_session.set_providers(["TensorrtExecutionProvider"],
    #                   [{'device_id': '1', 'trt_max_workspace_size': '2147483648', 'trt_fp16_enable': 'True'}])
    print("execuation providers:")
    print(ort_session.get_providers())
    # compute ONNX Runtime output prediction
    """warmup (the first one running latency is quite large for the onnx model)"""
    for i in range(50):
        # pytorch inference
        img_x_cuda, mask_x_cuda, feat_vec_z_cuda, mask_vec_z_cuda, pos_vec_z_cuda = \
            img_x.cuda(), mask_x.cuda(), feat_vec_z.cuda(), mask_vec_z.cuda(), pos_vec_z.cuda()
        torch_outs = torch_model(img_x_cuda, mask_x_cuda, feat_vec_z_cuda, mask_vec_z_cuda, pos_vec_z_cuda)
        # onnx inference
        ort_inputs = {'img_x': to_numpy(img_x),
                      'mask_x': to_numpy(mask_x),
                      'feat_vec_z': to_numpy(feat_vec_z),
                      'mask_vec_z': to_numpy(mask_vec_z),
                      'pos_vec_z': to_numpy(pos_vec_z)
                      }
        s_ort = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
    """begin the timing"""
    t_pyt = 0  # pytorch time
    t_ort = 0  # onnxruntime time

    for i in range(N):
        # generate data
        img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z = get_data(bs=bs, sz_x=sz_x, hw_z=hw_z, c=c)
        # pytorch inference
        img_x_cuda, mask_x_cuda, feat_vec_z_cuda, mask_vec_z_cuda, pos_vec_z_cuda = \
            img_x.cuda(), mask_x.cuda(), feat_vec_z.cuda(), mask_vec_z.cuda(), pos_vec_z.cuda()
        s_pyt = time.time()
        torch_outs = torch_model(img_x_cuda, mask_x_cuda, feat_vec_z_cuda, mask_vec_z_cuda, pos_vec_z_cuda)
        e_pyt = time.time()
        lat_pyt = e_pyt - s_pyt
        t_pyt += lat_pyt
        # print("pytorch latency: %.2fms" % (lat_pyt * 1000))
        # onnx inference
        ort_inputs = {'img_x': to_numpy(img_x),
                      'mask_x': to_numpy(mask_x),
                      'feat_vec_z': to_numpy(feat_vec_z),
                      'mask_vec_z': to_numpy(mask_vec_z),
                      'pos_vec_z': to_numpy(pos_vec_z)
                      }
        s_ort = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        e_ort = time.time()
        lat_ort = e_ort - s_ort
        t_ort += lat_ort
        # print("onnxruntime latency: %.2fms" % (lat_ort * 1000))
    print("pytorch model average latency", t_pyt/N*1000)
    print("onnx model average latency:", t_ort/N*1000)

    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_outs), ort_outs[0], rtol=1e-03, atol=1e-05)
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")
