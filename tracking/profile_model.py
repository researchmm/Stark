import argparse
import torch
import _init_paths
from lib.utils.merge import merge_template_search
# from lib.config.stark_s.config import cfg, update_config_from_file
# from lib.models.stark.stark_s import build_starks
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='stark_st2', choices=['stark_s', 'stark_st2'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline_R101', help='yaml configure file name')
    args = parser.parse_args()

    return args


def get_complexity_MHA(m:nn.MultiheadAttention, x, y):
    """(L, B, D): sequence length, batch size, dimension"""
    d_mid = m.embed_dim
    query, key, value = x[0], x[1], x[2]
    Lq, batch, d_inp = query.size()
    Lk = key.size(0)
    """compute flops"""
    total_ops = 0
    # projection of Q, K, V
    total_ops += d_inp * d_mid * Lq * batch  # query
    total_ops += d_inp * d_mid * Lk * batch * 2  # key and value
    # compute attention
    total_ops += Lq * Lk * d_mid * 2
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def evaluate(model, search, seq_dict, run_box_head, run_cls_head):
    """Compute FLOPs, Params, and Speed"""
    custom_ops = {nn.MultiheadAttention: get_complexity_MHA}
    # # backbone
    macs1, params1 = profile(model, inputs=(search, None, "backbone", False, False),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('backbone (search) macs is ', macs)
    print('backbone params is ', params)
    # transformer and head
    macs2, params2 = profile(model, inputs=(None, seq_dict, "transformer", True, True),
                             custom_ops=custom_ops, verbose=False)
    macs, params = clever_format([macs2, params2], "%.3f")
    print('transformer and head macs is ', macs)
    print('transformer and head params is ', params)
    # the whole model
    macs, params = clever_format([macs1 + macs2, params1 + params2], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    '''Speed Test'''
    T_w = 10
    T_t = 100
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(search, None, "backbone", run_box_head, run_cls_head)
            _ = model(None, seq_dict, "transformer", run_box_head, run_cls_head)
        start = time.time()
        for i in range(T_t):
            _ = model(search, None, "backbone", run_box_head, run_cls_head)
            _ = model(None, seq_dict, "transformer", run_box_head, run_cls_head)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        # backbone
        for i in range(T_w):
            _ = model(search, None, "backbone", run_box_head, run_cls_head)
        start = time.time()
        for i in range(T_t):
            _ = model(search, None, "backbone", run_box_head, run_cls_head)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average backbone latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    h_dim = cfg.MODEL.HIDDEN_DIM
    '''import stark network module'''
    model_module = importlib.import_module('lib.models.stark')
    if args.script == "stark_s":
        model_constructor = model_module.build_starks
        model = model_constructor(cfg)
        # get the template and search
        template = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        # forward template and search
        oup_t = model.forward_backbone(template)
        oup_s = model.forward_backbone(search)
        seq_dict = merge_template_search([oup_t, oup_s])
        # evaluate the model properties
        evaluate(model, search, seq_dict, run_box_head=True, run_cls_head=False)
    elif args.script == "stark_st2":
        model_constructor = model_module.build_starkst
        model = model_constructor(cfg)
        # get the template and search
        template1 = get_data(bs, z_sz)
        template2 = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        # transfer to device
        model = model.to(device)
        template1 = template1.to(device)
        template2 = template2.to(device)
        search = search.to(device)
        # forward template and search
        oup_t1 = model.forward_backbone(template1)
        oup_t2 = model.forward_backbone(template2)
        oup_s = model.forward_backbone(search)
        seq_dict = merge_template_search([oup_t1, oup_t2, oup_s])
        # evaluate the model properties
        evaluate(model, search, seq_dict, run_box_head=True, run_cls_head=True)
