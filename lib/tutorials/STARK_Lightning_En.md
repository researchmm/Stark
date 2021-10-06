# STARK-Lightning Tutorial
**Introduction**：[ONNXRUNTIME](https://github.com/microsoft/onnxruntime) is an open-source library by Microsoft for network inference acceleration. In this tutorial, we will show how to export the trained model to ONNX format 
and use ONNXRUNTIME to further accelerate the inference. The accelerated STARK-Lightning can run at 200+ FPS on a RTX TITAN GPU! let's get started. 
## STARK-Lightning v.s Other Trackers
| Tracker | LaSOT (AUC)| Speed (FPS) | Params (MB)|
|---|---|---|---|
|**STARK-Lightning**|**58.2**|**~200**|**8.2**|
|DiMP50|56.8|~50|165|
|DaSiamRPN|41.5|~200|362|
|SiamFC|33.6|~100|8.9|

STARK-Lightning achieves better performance than DiMP50, runs at a competitive speed as DaSiamRPN :zap: , and has a smaller model size than SiamFC!
## (Optionally) Train STARK-Lightning
Train STARK-Lightning with 8 GPUs with the following command
```
python tracking/train.py --script stark_lightning_X_trt --config baseline_rephead_4_lite_search5 --save_dir . --mode multiple --nproc_per_node 8
```
Since the training of STARK-Lightning is fast and memory-friendly, you can also train it with less GPUs (such as 2 or 4) by set ```nproc_per_node``` accordingly.  
## Install onnx and onnxruntime
for inference on GPU
```
pip install onnx onnxruntime-gpu==1.6.0
```
- Here the version of onnxruntime-gpu needs to be compatible to the CUDA version and CUDNN version on the machine. For more details, please refer to https://www.onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html
  . For example, on my computer, CUDA version is 10.2, CUDNN version is 8.0.3, so I choose onnxruntime-gpu==1.6.0 

for inference only on CPU
```
pip install onnx onnxruntime
```
## ONNX Conversion and Inference
Download trained PyTorch checkpoints [STARK_Lightning](https://drive.google.com/file/d/1dVme6p-_j0fFcxYQ-rrF07pRuf57uPA-/view?usp=sharing)

Export the trained PyTorch model to onnx format, then test it with onnxruntime
```
python tracking/ORT_lightning_X_trt_backbone_bottleneck_pe.py  # for the template branch
python tracking/ORT_lightning_X_trt_complete.py  # for the search region branch
```
- The conversion can run successfully in the terminal. However, it leads to an error of "libcudnn8.so is not found" when running in Pycharm. 
  So please run these two commands in the terminal.

Evaluate the converted onnx model on LaSOT (Support multiple-GPU inference).
- Set ```use_onnx=True``` in lib/test/tracker/stark_lightning_X_trt.py, then run
```
python tracking/test.py stark_lightning_X_trt baseline_rephead_4_lite_search5 --threads 8 --num_gpus 2
```
```num_gpus``` is the the number of GPUs to use，```threads``` is the number of processes. we usually set ```threads``` to be four times ```num_gpus```.
If the user want to run the sequences one by one, you can run the following command
```
python tracking/test.py stark_lightning_X_trt baseline_rephead_4_lite_search5 --threads 0 --num_gpus 1
```
- Evaluate the tracking results
```
python tracking/analysis_results_ITP.py --script stark_lightning_X_trt --config baseline_rephead_4_lite_search5
```
