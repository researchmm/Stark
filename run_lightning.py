import os
import sys
import argparse
import cv2
from easydict import EasyDict as edict
from PIL import Image
import yaml
import numpy as np
from collections import OrderedDict
import onnxruntime
import time
import math
import tensorflow as tf
from onnx2keras import onnx_to_keras
from onnx_tf.backend import prepare
import onnx

prj_dir = r'C:\Users\zadorozhnyy.v\Downloads\mystark'
save_dir = r'C:\Users\zadorozhnyy.v\Downloads\mystark\SL_model'    
data_dir = r'C:\Users\zadorozhnyy.v\Downloads\Stark-main\data\got10k\test'

def process(img_arr: np.ndarray, amask_arr: np.ndarray):
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
    """img_arr: (H,W,3), amask_arr: (H,W)"""
    # Deal with the image patch
    img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    img_arr_4d = (img_arr_4d / 255.0 - mean) / std  # (1, 3, H, W)
    # Deal with the attention mask
    amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
    return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)

def sample_target(im, target_bb, search_area_factor, output_sz=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        return im_crop_padded, resize_factor, att_mask

    else:
        return im_crop_padded, att_mask.astype(np.bool_), 1.0

def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h] 

def map_box_back(state, pred_box: list, search_size, resize_factor: float):
    cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def save_res(im_dir, data):
    print(im_dir.split('\\')[-1])
    file = os.path.join(prj_dir,"lib\\test\\tracking_results\\stark_lightning_X_trt\\baseline_rephead_4_lite_search5\\got10k", im_dir.split('\\')[-1] + ".txt")
    print(file)
    tracked_bb = np.array(data).astype(int)
    np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')    

def get_new_frame(frame_id, im_dir):
    imgs = [img for img in os.listdir(im_dir) if img.endswith(".jpg")]
    if len(imgs) <= frame_id:
        return None
    im = cv2.imread(os.path.join(im_dir, imgs[frame_id]))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def get_init_box(im_dir):
    path = os.path.join(im_dir, "groundtruth.txt")
    ground_truth_rect = np.loadtxt(path, delimiter=',', dtype=np.float64)
    return ground_truth_rect

def my_tracker(get_new_frame, get_init_box, im_dir, backend):
    params = edict()
    # template and search region
    params.template_factor = 2.0
    params.template_size = 128
    params.search_factor = 5.0
    params.search_size = 320
    if backend == 'onnx':
        gpu_id = 0
        providers = ["CUDAExecutionProvider"]
        provider_options = [{"device_id": str(gpu_id)}]
        ort_sess_z = onnxruntime.InferenceSession(os.path.join(save_dir, "backbone_bottleneck_pe.onnx"), providers=providers,
                                                       provider_options=provider_options)
        ort_sess_x = onnxruntime.InferenceSession(os.path.join(save_dir, "complete.onnx"), providers=providers,
                                                       provider_options=provider_options)
    else:

        ort_sess_z = tf.saved_model.load(os.path.join(save_dir, "tf_backbone"))
        #onnx_model = onnx.load(os.path.join(save_dir, "backbone_bottleneck_pe.onnx"))
        ort_sess_x = tf.saved_model.load(os.path.join(save_dir, "tf_complete"))
        
    frame_id = 0
    ort_outs_z = []

    state = get_init_box(im_dir)
    image = get_new_frame(frame_id, im_dir)
    z_patch_arr, _, z_amask_arr = sample_target(image, state, params.template_factor, output_sz=params.template_size)
    #print(z_patch_arr)
    template, template_mask = process(z_patch_arr, z_amask_arr)
    # forward the template once
    ort_inputs = {'img_z': template, 'mask_z': template_mask}
    if backend == 'onnx':
        ort_outs_z = ort_sess_z.run(None, ort_inputs)
    else:
        #ort_outs_z = ort_sess_z(template, template_mask)
        ort_outs_z = ort_sess_z(img_z = template, mask_z = template_mask)
    outputs = []
    outputs.append(state)
    frame_id = 1
    image = get_new_frame(frame_id, im_dir)
    while image is not None:
        H, W, _ = image.shape
        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, state, params.search_factor,
                                                                output_sz=params.search_size)  # (x1, y1, w, h)
        search, search_mask = process(x_patch_arr, x_amask_arr)
        
        if backend == 'onnx':
            ort_inputs = {'img_x': search,
                      'mask_x': search_mask,
                      'feat_vec_z': ort_outs_z[0],
                      'mask_vec_z': ort_outs_z[1],
                      'pos_vec_z': ort_outs_z[2],
                      }
        
            ort_outs = ort_sess_x.run(None, ort_inputs)
            pred_box = (ort_outs[0].reshape(4) * params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        else:

            ort_outs = ort_sess_x(img_x = search, mask_x = search_mask, feat_vec_z = ort_outs_z['feat'], mask_vec_z = ort_outs_z['mask'], pos_vec_z =  ort_outs_z['pos'] )
            coords = ort_outs['outputs_coord'].numpy()[0]
            pred_box = (coords * params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        state = clip_box(map_box_back(state, pred_box, params.search_size, resize_factor), H, W, margin=10)

        outputs.append(state)
        frame_id += 1
        image = get_new_frame(frame_id, im_dir)
    return outputs
'''
Script arguments: 
1) Path to the folder - full path to the directory with images with .jpg extension; image names are numbers in increasing order.
Directory should also contain the file groundtruth.txt with the position of the object at the initial 1st frame.
2) backend: onnx or tensorflow
'''
def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--folder_path', type=str, default=None, help='Path to the folder')
    parser.add_argument('--backend', type=str, default='tf')
    args = parser.parse_args()
    outputs = my_tracker(get_new_frame, get_init_box, args.folder_path, args.backend)
    save_res(args.folder_path, outputs)
if __name__ == '__main__':
    main()