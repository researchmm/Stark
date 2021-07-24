import argparse
import torch
import _init_paths
from lib.utils.merge import get_qkv
from thop import profile
from thop.utils import clever_format
import time
from lib.models.stark.repvgg import repvgg_model_convert
from lib.models.stark import build_stark_lightning_x_trt
from lib.config.stark_lightning_X_trt.config import cfg, update_config_from_file


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='stark_lightning_X_trt',
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline_rephead_4', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate(model, img_x, att_x, q, k, v, key_padding_mask):
    """Compute FLOPs, Params, and Speed"""
    # backbone
    macs1, params1 = profile(model, inputs=(img_x, att_x, None, None, None, None, "backbone", "search"),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('backbone (search) macs is ', macs)
    print('backbone params is ', params)
    # transformer and head
    macs2, params2 = profile(model, inputs=(None, None, q, k, v, key_padding_mask, "transformer", "search"),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs2, params2], "%.3f")
    print('transformer and head macs is ', macs)
    print('transformer and head params is ', params)
    # the whole model
    macs, params = clever_format([macs1 + macs2, params1 + params2], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)


def get_data(bs, sz, hw_z=64, hw_x=256, c=256):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    q = torch.randn(hw_x, bs, c)
    k = torch.randn(hw_z + hw_x, bs, c)
    v = torch.randn(hw_z + hw_x, bs, c)
    key_padding_mask = torch.rand(bs, hw_z + hw_x) > 0.5
    return img_patch, att_mask, q, k, v, key_padding_mask


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # device = "cpu"
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    hw_z = cfg.DATA.TEMPLATE.FEAT_SIZE ** 2
    hw_x = cfg.DATA.SEARCH.FEAT_SIZE ** 2
    c = cfg.MODEL.HIDDEN_DIM
    # build the stark model
    model = build_stark_lightning_x_trt(cfg, phase='test')
    # transfer to test mode
    model = repvgg_model_convert(model)
    model.eval()
    print(model)
    # get the input data
    img_x, att_x, q, k, v, key_padding_mask = get_data(bs, x_sz, hw_z, hw_x, c)
    # transfer to device
    model = model.to(device)
    model.box_head.coord_x = model.box_head.coord_x.to(device)
    model.box_head.coord_y = model.box_head.coord_y.to(device)
    img_x, att_x, q, k, v, key_padding_mask = img_x.to(device), att_x.to(device), \
                                              q.to(device), k.to(device), v.to(device), key_padding_mask.to(device)

    # evaluate the model properties
    evaluate(model, img_x, att_x, q, k, v, key_padding_mask)
    '''Speed Test'''
    T_w = 50  # warmup time
    T_t = 1000  # test time
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(img=img_x, mask=att_x, mode="backbone", zx="search")
            _ = model(q=q, k=k, v=v, key_padding_mask=key_padding_mask, mode="transformer")
        t_all = 0  # overall latency
        t_back = 0  # backbone latency
        for i in range(T_t):
            # get the input data
            img_x, att_x, q, k, v, key_padding_mask = get_data(bs, x_sz, hw_z, hw_x, c)
            img_x, att_x, q, k, v, key_padding_mask = img_x.to(device), att_x.to(device), \
                                                      q.to(device), k.to(device), v.to(device), \
                                                      key_padding_mask.to(device)
            s = time.time()
            _ = model(img=img_x, mask=att_x, mode="backbone", zx="search")
            e_b = time.time()
            _ = model(q=q, k=k, v=v, key_padding_mask=key_padding_mask, mode="transformer")
            e = time.time()
            lat = e - s
            lat_b = e_b - s
            t_all += lat
            t_back += lat_b
            # print("backbone latency: %.2fms, overall latency: %.2fms" % (lat_b*1000, lat*1000))
        print("The average overall latency is %.2f ms" % (t_all/T_t * 1000))
        print("The average backbone latency is %.2f ms" % (t_back/T_t * 1000))

