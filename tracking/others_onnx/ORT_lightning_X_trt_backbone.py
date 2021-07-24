import argparse
import torch
import _init_paths
from lib.models.stark.repvgg import repvgg_model_convert
from lib.models.stark import build_stark_lightning_x_trt
from lib.config.stark_lightning_X_trt.config import cfg, update_config_from_file
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


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz, requires_grad=True)
    return img_patch


if __name__ == "__main__":
    save_name = "backbone.onnx"
    """update cfg"""
    args = parse_args()
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    x_sz = cfg.TEST.SEARCH_SIZE
    # build the stark model
    model = build_stark_lightning_x_trt(cfg, phase='test')
    # transfer to test mode
    model = repvgg_model_convert(model)
    model.eval()
    """in this script, we only test the backbone"""
    torch_model = model.backbone
    print(torch_model)
    # get the template and search
    img_x = get_data(bs, x_sz)
    # forward the search region
    oup_s = torch_model(img_x)
    torch.onnx.export(torch_model,  # model being run
                      img_x,  # model input (or a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(save_name)


    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(oup_s), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
