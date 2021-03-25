from lib.test.vot20.stark_vot20 import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
run_vot_exp('stark_st', 'baseline_R101', vis=False)
