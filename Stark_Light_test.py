import os
import sys
import argparse
import cv2
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
from easydict import EasyDict as edict
from PIL import Image
import yaml
import numpy as np
from collections import OrderedDict
import onnxruntime
import time
import math

class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)
    
def load_text(path, delimiter, dtype):
        if isinstance(delimiter, (tuple, list)):
            for d in delimiter:
                try:
                    ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                    return ground_truth_rect
                except:
                    pass

            raise Exception('Could not read file {}'.format(path))
        else:
            ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
            return ground_truth_rect
def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
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
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

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
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded  

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

class EnvSettings:
    def __init__(self):
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.results_path = '{}/tracking_results/'.format(test_path)
        self.segmentation_path = '{}/segmentation_results/'.format(test_path)
        self.network_path = '{}/networks/'.format(test_path)
        self.result_plot_path = '{}/result_plots/'.format(test_path)
        self.got10k_path = r'C:\Users\zadorozhnyy.v\Downloads\Stark-main\data\got10k'
        self.network_path = r'C:\Users\zadorozhnyy.v\Downloads\mystark\lib\test/networks/'    # Where tracking networks are stored.
        self.prj_dir = r'C:\Users\zadorozhnyy.v\Downloads\mystark'
        self.save_dir = r'C:\Users\zadorozhnyy.v\Downloads\mystark'
        self.result_plot_path = r'C:\Users\zadorozhnyy.v\Downloads\mystark\lib\test/result_plots/'
        self.results_path = r'C:\Users\zadorozhnyy.v\Downloads\mystark\lib\test/tracking_results/'    # Where to store tracking results
        self.segmentation_path = r'C:\Users\zadorozhnyy.v\Downloads\mystark\lib\test/segmentation_results/'

class Sequence:
    """Class for the sequence in an evaluation."""
    def __init__(self, name, frames, dataset, ground_truth_rect, ground_truth_seg=None, init_data=None,
                 object_class=None, target_visible=None, object_ids=None, multiobj_mode=False):
        self.name = name
        self.frames = frames
        self.dataset = dataset
        self.ground_truth_rect = ground_truth_rect
        self.ground_truth_seg = ground_truth_seg
        self.object_class = object_class
        self.target_visible = target_visible
        self.object_ids = object_ids
        self.multiobj_mode = multiobj_mode
        self.init_data = self._construct_init_data(init_data)
        self._ensure_start_frame()

    def _ensure_start_frame(self):
        # Ensure start frame is 0
        start_frame = min(list(self.init_data.keys()))
        if start_frame > 0:
            self.frames = self.frames[start_frame:]
            if self.ground_truth_rect is not None:
                if isinstance(self.ground_truth_rect, (dict, OrderedDict)):
                    for obj_id, gt in self.ground_truth_rect.items():
                        self.ground_truth_rect[obj_id] = gt[start_frame:,:]
                else:
                    self.ground_truth_rect = self.ground_truth_rect[start_frame:,:]
            if self.ground_truth_seg is not None:
                self.ground_truth_seg = self.ground_truth_seg[start_frame:]
                assert len(self.frames) == len(self.ground_truth_seg)

            if self.target_visible is not None:
                self.target_visible = self.target_visible[start_frame:]
            self.init_data = {frame-start_frame: val for frame, val in self.init_data.items()}

    def _construct_init_data(self, init_data):
        if init_data is not None:
            if not self.multiobj_mode:
                assert self.object_ids is None or len(self.object_ids) == 1
                for frame, init_val in init_data.items():
                    if 'bbox' in init_val and isinstance(init_val['bbox'], (dict, OrderedDict)):
                        init_val['bbox'] = init_val['bbox'][self.object_ids[0]]
            # convert to list
            for frame, init_val in init_data.items():
                if 'bbox' in init_val:
                    if isinstance(init_val['bbox'], (dict, OrderedDict)):
                        init_val['bbox'] = OrderedDict({obj_id: list(init) for obj_id, init in init_val['bbox'].items()})
                    else:
                        init_val['bbox'] = list(init_val['bbox'])
        else:
            init_data = {0: dict()}     # Assume start from frame 0

            if self.object_ids is not None:
                init_data[0]['object_ids'] = self.object_ids

            if self.ground_truth_rect is not None:
                if self.multiobj_mode:
                    assert isinstance(self.ground_truth_rect, (dict, OrderedDict))
                    init_data[0]['bbox'] = OrderedDict({obj_id: list(gt[0,:]) for obj_id, gt in self.ground_truth_rect.items()})
                else:
                    assert self.object_ids is None or len(self.object_ids) == 1
                    if isinstance(self.ground_truth_rect, (dict, OrderedDict)):
                        init_data[0]['bbox'] = list(self.ground_truth_rect[self.object_ids[0]][0, :])
                    else:
                        init_data[0]['bbox'] = list(self.ground_truth_rect[0,:])

            if self.ground_truth_seg is not None:
                init_data[0]['mask'] = self.ground_truth_seg[0]

        return init_data

    def init_info(self):
        info = self.frame_info(frame_num=0)
        return info

    def frame_info(self, frame_num):
        info = self.object_init_data(frame_num=frame_num)
        return info

    def init_bbox(self, frame_num=0):
        return self.object_init_data(frame_num=frame_num).get('init_bbox')

    def init_mask(self, frame_num=0):
        return self.object_init_data(frame_num=frame_num).get('init_mask')

    def get_info(self, keys, frame_num=None):
        info = dict()
        for k in keys:
            val = self.get(k, frame_num=frame_num)
            if val is not None:
                info[k] = val
        return info

    def object_init_data(self, frame_num=None) -> dict:
        if frame_num is None:
            frame_num = 0
        if frame_num not in self.init_data:
            return dict()

        init_data = dict()
        for key, val in self.init_data[frame_num].items():
            if val is None:
                continue
            init_data['init_'+key] = val

        if 'init_mask' in init_data and init_data['init_mask'] is not None:
            im = Image.open(init_data['init_mask'])
            anno = np.atleast_3d(im)[...,0]
            if not self.multiobj_mode and self.object_ids is not None:
                assert len(self.object_ids) == 1
                anno = (anno == int(self.object_ids[0])).astype(np.uint8)
            init_data['init_mask'] = anno

        if self.object_ids is not None:
            init_data['object_ids'] = self.object_ids
            init_data['sequence_object_ids'] = self.object_ids

        return init_data

    def target_class(self, frame_num=None):
        return self.object_class

    def get(self, name, frame_num=None):
        return getattr(self, name)(frame_num)

    def __repr__(self):
        return "{self.__class__.__name__} {self.name}, length={len} frames".format(self=self, len=len(self.frames))



class SequenceList(list):
    """List of sequences. Supports the addition operator to concatenate sequence lists."""
    def __getitem__(self, item):
        if isinstance(item, str):
            for seq in self:
                if seq.name == item:
                    return seq
            raise IndexError('Sequence name not in the dataset.')
        elif isinstance(item, int):
            return super(SequenceList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return SequenceList([super(SequenceList, self).__getitem__(i) for i in item])
        else:
            return SequenceList(super(SequenceList, self).__getitem__(item))

    def __add__(self, other):
        return SequenceList(super(SequenceList, self).__add__(other))

    def copy(self):
        return SequenceList(super(SequenceList, self).copy())

class GOT10KDataset:
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split):
        self.env_settings = EnvSettings()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(self.env_settings.got10k_path, split)
        else:
            self.base_path = os.path.join(self.env_settings.got10k_path, 'train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split
    
    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'got10k', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list
    
class TrackerParams:
    """Class for tracker parameters."""
    def set_default_values(self, default_vals: dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)

    def get(self, name: str, *default):
        """Get a parameter value with the given name. If it does not exists, it return the default value given as a
        second argument or returns an error if no default value is given."""
        if len(default) > 1:
            raise ValueError('Can only give one default value.')

        if not default:
            return getattr(self, name)

        return getattr(self, name, default[0])

    def has(self, name: str):
        """Check if there exist a parameter with the given name."""
        return hasattr(self, name)
    
class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = EnvSettings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()
        print("parameters = ")
        print(params)
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()
        print("tracker.run_sequence: init_info")
        print(init_info)
        tracker = STARK_LightningXtrt_onnx(params, self.dataset_name)
        print("after params")
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output


    def get_parameters(self):
        
        def _edict2dict(dest_dict, src_edict):
            if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
                for k, v in src_edict.items():
                    if not isinstance(v, edict):
                        dest_dict[k] = v
                    else:
                        dest_dict[k] = {}
                        _edict2dict(dest_dict[k], v)
            else:
                return

        def gen_config(config_file):
            cfg_dict = {}
            _edict2dict(cfg_dict, cfg)
            with open(config_file, 'w') as f:
                yaml.dump(cfg_dict, f, default_flow_style=False)


        def _update_config(base_cfg, exp_cfg):
            if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
                for k, v in exp_cfg.items():
                    if k in base_cfg:
                        if not isinstance(v, dict):
                            base_cfg[k] = v
                        else:
                            _update_config(base_cfg[k], v)
                    else:
                        raise ValueError("{} not exist in config.py".format(k))
            else:
                return

        def update_config_from_file(filename):
            exp_config = None
            with open(filename) as f:
                exp_config = edict(yaml.safe_load(f))
                _update_config(cfg, exp_config)

        """Get parameters."""
        params = TrackerParams()
        prj_dir = EnvSettings().prj_dir
        save_dir = EnvSettings().save_dir
        cfg = edict()

        # MODEL
        cfg.MODEL = edict()
        cfg.MODEL.HEAD_TYPE = "CORNER_LITE"
        cfg.MODEL.HIDDEN_DIM = 256
        cfg.MODEL.HEAD_DIM = 256  # channel in the corner head
        # MODEL.BACKBONE
        cfg.MODEL.BACKBONE = edict()
        cfg.MODEL.BACKBONE.TYPE = "RepVGG-A0"  # resnet50, resnext101_32x8d
        cfg.MODEL.BACKBONE.OUTPUT_LAYERS = ["stage3"]
        cfg.MODEL.BACKBONE.DILATION = False
        cfg.MODEL.BACKBONE.LAST_STAGE_BLOCK = 14
        # MODEL.TRANSFORMER
        cfg.MODEL.TRANSFORMER = edict()
        cfg.MODEL.TRANSFORMER.NHEADS = 8
        cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
        cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 2048

        # TRAIN
        cfg.TRAIN = edict()
        cfg.TRAIN.DISTILL = False
        cfg.TRAIN.DISTILL_LOSS_TYPE = "KL"
        cfg.TRAIN.AMP = False
        cfg.TRAIN.LR = 0.0001
        cfg.TRAIN.WEIGHT_DECAY = 0.0001
        cfg.TRAIN.EPOCH = 500
        cfg.TRAIN.LR_DROP_EPOCH = 400
        cfg.TRAIN.BATCH_SIZE = 16
        cfg.TRAIN.NUM_WORKER = 8
        cfg.TRAIN.OPTIMIZER = "ADAMW"
        cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
        cfg.TRAIN.GIOU_WEIGHT = 2.0
        cfg.TRAIN.L1_WEIGHT = 5.0
        cfg.TRAIN.DEEP_SUPERVISION = False
        cfg.TRAIN.FREEZE_BACKBONE_BN = True
        cfg.TRAIN.BACKBONE_TRAINED_LAYERS = ['stage2', 'stage3']
        cfg.TRAIN.PRINT_INTERVAL = 50
        cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
        cfg.TRAIN.GRAD_CLIP_NORM = 0.1
        # TRAIN.SCHEDULER
        cfg.TRAIN.SCHEDULER = edict()
        cfg.TRAIN.SCHEDULER.TYPE = "step"
        cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

        # DATA
        cfg.DATA = edict()
        cfg.DATA.MEAN = [0.485, 0.456, 0.406]
        cfg.DATA.STD = [0.229, 0.224, 0.225]
        cfg.DATA.MAX_SAMPLE_INTERVAL = 200
        # DATA.TRAIN
        cfg.DATA.TRAIN = edict()
        cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
        cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
        cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
        # DATA.VAL
        cfg.DATA.VAL = edict()
        cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
        cfg.DATA.VAL.DATASETS_RATIO = [1]
        cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
        # DATA.SEARCH
        cfg.DATA.SEARCH = edict()
        cfg.DATA.SEARCH.SIZE = 320
        cfg.DATA.SEARCH.FEAT_SIZE = 20
        cfg.DATA.SEARCH.FACTOR = 5.0
        cfg.DATA.SEARCH.CENTER_JITTER = 4.5
        cfg.DATA.SEARCH.SCALE_JITTER = 0.5
        # DATA.TEMPLATE
        cfg.DATA.TEMPLATE = edict()
        cfg.DATA.TEMPLATE.SIZE = 128
        cfg.DATA.TEMPLATE.FEAT_SIZE = 8
        cfg.DATA.TEMPLATE.FACTOR = 2.0
        cfg.DATA.TEMPLATE.CENTER_JITTER = 0
        cfg.DATA.TEMPLATE.SCALE_JITTER = 0

        # TEST
        cfg.TEST = edict()
        cfg.TEST.TEMPLATE_FACTOR = 2.0
        cfg.TEST.TEMPLATE_SIZE = 128
        cfg.TEST.SEARCH_FACTOR = 5.0
        cfg.TEST.SEARCH_SIZE = 320
        cfg.TEST.EPOCH = 500
        yaml_name = "baseline_rephead_4_lite_search5"
        # update default config from yaml file
        yaml_file = os.path.join(prj_dir, 'experiments/stark_lightning_X_trt/%s.yaml' % yaml_name)

        update_config_from_file(yaml_file)
        params.cfg = cfg
        print("test config: ", cfg)

        # template and search region
        params.template_factor = cfg.TEST.TEMPLATE_FACTOR
        params.template_size = cfg.TEST.TEMPLATE_SIZE
        params.search_factor = cfg.TEST.SEARCH_FACTOR
        params.search_size = cfg.TEST.SEARCH_SIZE

        # Network checkpoint path
        params.checkpoint = os.path.join(save_dir,
                                         "checkpoints/train/stark_lightning_X_trt/%s/STARKLightningXtrt_ep%04d.pth.tar" %
                                         (yaml_name, cfg.TEST.EPOCH))
        # whether to save boxes from all queries
        params.save_all_boxes = False
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv2.imread(image_file)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")

class STARK_LightningXtrt_onnx:
    def __init__(self, params, dataset_name):
        self.params = params
        self.visdom = None
        """build two sessions"""
        '''2021.7.5 Add multiple gpu support'''
        num_gpu = 2
        try:
            worker_name = multiprocessing.current_process().name
            worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
            gpu_id = worker_id % num_gpu
            print(gpu_id)
        except:
            gpu_id = 0
        providers = ["CUDAExecutionProvider"]
        provider_options = [{"device_id": str(gpu_id)}]
        self.ort_sess_z = onnxruntime.InferenceSession("backbone_bottleneck_pe.onnx", providers=providers,
                                                       provider_options=provider_options)
        self.ort_sess_x = onnxruntime.InferenceSession("complete.onnx", providers=providers,
                                                       provider_options=provider_options)
        self.preprocessor = PreprocessorX_onnx()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        self.save_dir = "debug"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.ort_outs_z = []

    def initialize(self, image, info: dict):

        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        print(z_patch_arr.shape)
        template, template_mask = self.preprocessor.process(z_patch_arr, z_amask_arr)
        # forward the template once
        ort_inputs = {'img_z': template, 'mask_z': template_mask}
        self.ort_outs_z = self.ort_sess_z.run(None, ort_inputs)
        print(self.ort_outs_z)
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search, search_mask = self.preprocessor.process(x_patch_arr, x_amask_arr)

        ort_inputs = {'img_x': search,
                      'mask_x': search_mask,
                      'feat_vec_z': self.ort_outs_z[0],
                      'mask_vec_z': self.ort_outs_z[1],
                      'pos_vec_z': self.ort_outs_z[2],
                      }

        ort_outs = self.ort_sess_x.run(None, ort_inputs)
        if self.frame_id == 1:
            print(ort_outs[0].reshape(4))
        pred_box = (ort_outs[0].reshape(4) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug

        x1, y1, w, h = self.state
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
        save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
        cv2.imwrite(save_path, image_BGR)
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
        os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')


    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)
    
def run_tracker(tracker_name, tracker_param, run_id=None, sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = GOT10KDataset(split='test').get_sequence_list()

    if sequence is not None:
        dataset = [dataset[sequence]]

    tracker = Tracker(tracker_name, tracker_param, 'got10k_test', run_id)

    for seq in dataset:
        try:
            output = tracker.run_sequence(seq, debug=debug)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))
    _save_tracker_output(seq, tracker, output)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()
