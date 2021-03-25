import argparse
import torch
import _init_paths
from lib.utils.merge import merge_template_search
from lib.config.stark_st2.config import cfg, update_config_from_file
from lib.models.stark.stark_st import build_starkst
from tracking.profile_model import evaluate, get_data


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='stark_st2', help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    device = "cuda:7"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK model
    '''build the model'''
    args = parse_args()
    cfg_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(cfg_fname)
    model = build_starkst(cfg)
    '''get toy data'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    h_dim = cfg.MODEL.HIDDEN_DIM
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
