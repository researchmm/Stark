import argparse
import _init_paths
# for onnx conversion and inference
import numpy as np
import onnx
import onnxruntime
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--script', type=str, default='stark_lightning_X_trt', help='script name')
    parser.add_argument('--config', type=str, default='baseline_rephead', help='yaml configure file name')
    args = parser.parse_args()
    return args


def get_data(bs=1, sz_x=256, hw_z=64, c=256):
    img_x = np.random.randn(bs, 3, sz_x, sz_x).astype(np.float32)
    mask_x = np.random.rand(bs, sz_x, sz_x).astype(np.float32) > 0.5
    feat_vec_z = np.random.randn(hw_z, bs, c).astype(np.float32)  # HWxBxC
    mask_vec_z = np.random.rand(bs, hw_z).astype(np.float32) > 0.5  # BxHW
    pos_vec_z = np.random.randn(hw_z, bs, c).astype(np.float32)  # HWxBxC
    return img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z


# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    save_name = "complete.onnx"
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
        # onnx inference
        img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z = get_data()
        ort_inputs = {'img_x': img_x,
                      'mask_x': mask_x,
                      'feat_vec_z': feat_vec_z,
                      'mask_vec_z': mask_vec_z,
                      'pos_vec_z': pos_vec_z
                      }
        ort_outs = ort_session.run(None, ort_inputs)
    """timing"""
    N = 1000
    t_ort = 0
    for i in range(N):
        # onnx inference
        img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z = get_data()
        ort_inputs = {'img_x': img_x,
                      'mask_x': mask_x,
                      'feat_vec_z': feat_vec_z,
                      'mask_vec_z': mask_vec_z,
                      'pos_vec_z': pos_vec_z
                      }
        s_ort = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        e_ort = time.time()
        t_ort += e_ort - s_ort
    print("onnx model average latency:", t_ort/N*1000)