import argparse
import torch
import _init_paths
from lib.utils.merge import merge_template_search
from lib.config.stark_s.config import cfg, update_config_from_file
from lib.models.stark.stark_s import build_starks
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='stark_s', help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate(model, search, seq_dict, run_box_head, run_cls_head):
    '''Compute FLOPs, Params, and Speed'''
    # # backbone
    macs1, params1 = profile(model, inputs=(search, None, "backbone", False, False),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('backbone (search) macs is ', macs)
    print('backbone params is ', params)
    # transformer and head
    macs2, params2 = profile(model, inputs=(None, seq_dict, "transformer", True, True),
                             custom_ops=None, verbose=False)
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
    device = "cuda:7"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    '''build the model'''
    args = parse_args()
    cfg_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(cfg_fname)
    model = build_starks(cfg)
    '''get toy data'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    h_dim = cfg.MODEL.HIDDEN_DIM
    # get the template
    template = get_data(bs, z_sz)
    # get the search
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
    evaluate(model, search, seq_dict, run_box_head=False, run_cls_head=False)
