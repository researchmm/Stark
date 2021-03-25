from pytracking.VOT2020_super_only_mask_384_HP.dimp_alpha_seg_class import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# run_vot_exp('dimp','dimp50_vot19','SEbcm',0.60,VIS=False)
run_vot_exp('dimp','super_dimp','ARcm_coco_seg_only_mask_384',0.65,VIS=False)
# run_vot_exp('dimp','super_dimp','ARcm_coco_seg_only_mask_384',0.65,VIS=True)