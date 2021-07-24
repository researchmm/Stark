import argparse
import torch
import _init_paths
from lib.models.stark.repvgg import repvgg_model_convert
from lib.models.stark import build_stark_lightning_x_trt
from lib.config.stark_lightning_X_trt.config import cfg, update_config_from_file
from lib.utils.merge import get_qkv
import torch.onnx
import numpy as np
import onnx
import onnxruntime


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='stark_lightning_X_trt',
                        help='training script name')
    parser.add_argument('--config', type=str, default='baseline_rephead', help='yaml configure file name')
    args = parser.parse_args()

    return args


def get_data(bs, hw, c):
    feat_vec = torch.randn(hw, bs, c, requires_grad=True)  # HWxBxC
    pos_embed_vec = torch.randn(hw, bs, c, requires_grad=True)  # HWxBxC
    mask_vec = torch.rand(bs, hw, requires_grad=True) > 0.5  # BxHW
    return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}


if __name__ == "__main__":
    save_name = "transformer.onnx"
    """update cfg"""
    args = parse_args()
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    hw_x = cfg.DATA.SEARCH.FEAT_SIZE ** 2
    hw_z = cfg.DATA.TEMPLATE.FEAT_SIZE ** 2
    c = cfg.MODEL.HIDDEN_DIM
    # build the stark model
    model = build_stark_lightning_x_trt(cfg, phase='test')
    # transfer to test mode
    model = repvgg_model_convert(model)
    model.eval()
    """in this script, we only test the transformer"""
    model = model.transformer
    print(model)
    # get the template and search
    dict_z = get_data(bs, hw_z, c)
    dict_x = get_data(bs, hw_x, c)
    q, k, v, key_padding_mask = get_qkv([dict_z, dict_x])
    # forward the transformer
    oup_s = model(q, k, v, key_padding_mask)
    torch.onnx.export(model,  # model being run
                      (q, k, v, key_padding_mask),  # model input (or a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['q', 'k', 'v', 'key_padding_mask'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      )

    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(save_name)


    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    # compute ONNX Runtime output prediction
    ort_inputs = {'q': to_numpy(q),
                  'k': to_numpy(k),
                  'v': to_numpy(v),
                  'key_padding_mask': to_numpy(key_padding_mask)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(oup_s), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
