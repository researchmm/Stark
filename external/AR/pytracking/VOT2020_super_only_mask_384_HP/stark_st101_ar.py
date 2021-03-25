from pytracking.VOT2020_super_only_mask_384_HP.stark_alpha_seg_class import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
run_vot_exp('stark_st', 'baseline_R101',
            'ARcm_coco_seg_only_mask_384', 0.65, VIS=False)
