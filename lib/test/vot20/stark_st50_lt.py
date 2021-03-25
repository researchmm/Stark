from lib.test.vot20.stark_vot20lt import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
run_vot_exp('stark_st', 'baseline', vis=False)
