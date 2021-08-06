# OTB2015
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_s baseline --dataset otb --threads 28 --num_gpus 7
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_st baseline --dataset otb --threads 28 --num_gpus 7
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_st baseline_R101 --dataset otb --threads 28 --num_gpus 7
# UAV123
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_s baseline --dataset uav --threads 28 --num_gpus 7
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_st baseline --dataset uav --threads 28 --num_gpus 7
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_st baseline_R101 --dataset uav --threads 28 --num_gpus 7
# NFS
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_s baseline --dataset nfs --threads 28 --num_gpus 7
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_st baseline --dataset nfs --threads 28 --num_gpus 7
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_st baseline_R101 --dataset nfs --threads 28 --num_gpus 7
# TC128
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_s baseline --dataset tc128 --threads 28 --num_gpus 7
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_st baseline --dataset tc128 --threads 28 --num_gpus 7
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python tracking/test.py stark_st baseline_R101 --dataset tc128 --threads 28 --num_gpus 7

