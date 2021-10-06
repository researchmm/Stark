# STARK-Lightning 中文教程
**前言**： [ONNXRUNTIME](https://github.com/microsoft/onnxruntime) 是微软开源的一个用于网络推理加速的库，在该教程中我们将教给大家如何将训练好的模型导出成ONNX格式，
并使用ONNXRUNTIME来进一步加速推理，加速后的STARK-Lightning在RTX TITAN上的运行速度可达200+ FPS！让我们开始吧
## STARK-Lightning v.s 其他跟踪器
| Tracker | LaSOT (AUC)| Speed (FPS) | Params (MB)|
|---|---|---|---|
|**STARK-Lightning**|**58.2**|**~200**|**8.2**|
|DiMP50|56.8|~50|165|
|DaSiamRPN|41.5|~200|362|
|SiamFC|33.6|~100|8.9|

STARK-Lightning取得了比DiMP50更强的性能，运行速度和DaSiamRPN一样快 :zap: ，而模型大小比SiamFC还要更小一些！
## (非必须) 训练 STARK-Lightning
运行下面的指令，可8卡并行训练
```
python tracking/train.py --script stark_lightning_X_trt --config baseline_rephead_4_lite_search5 --save_dir . --mode multiple --nproc_per_node 8
```
由于STARK-Lightning的训练很快，并且只需要极少的显存，因此也可以考虑用2卡或者4卡训练，只需对应修改 ```nproc_per_node```即可.
## 安装onnx和onnxruntime
如果想在GPU上使用onnxruntime完成推理
```
pip install onnx onnxruntime-gpu==1.6.0
```
- 这里onnxruntime-gpu的版本需要和机器上的CUDA版本还有CUDNN版本适配，版本对应关系请参考https://www.onnxruntime.ai/docs/reference/execution-providers/CUDA-ExecutionProvider.html
。在我的电脑上，CUDA版本10.2，CUDNN版本8.0.3，故安装的是onnxruntime-gpu==1.6.0

如果只需要在CPU上使用
```
pip install onnx onnxruntime
```
## ONNX模型转换与推理测试
下载训练好的PyTorch模型权重文件 [STARK_Lightning](https://drive.google.com/file/d/1dVme6p-_j0fFcxYQ-rrF07pRuf57uPA-/view?usp=sharing)

将训练好的PyTorch模型转换成onnx格式,并测试onnxruntime
```
python tracking/ORT_lightning_X_trt_backbone_bottleneck_pe.py  # for the template branch
python tracking/ORT_lightning_X_trt_complete.py  # for the search region branch
```
- 模型转换在终端里可以跑通，但是在pycharm里面会报找不到libcudnn8.so的错误，后面就在终端运行吧

在LaSOT上测试转换后的模型（支持多卡推理）
- 首先在lib/test/tracker/stark_lightning_X_trt.py中设置 use_onnx = True, 之后运行
```
python tracking/test.py stark_lightning_X_trt baseline_rephead_4_lite_search5 --threads 8 --num_gpus 2
```
其中num_gpus是想使用的GPU数量，threads是进程数量，我们通常将其设置成GPU数量的4倍。
如果想一个一个视频来跑，可以运行以下指令
```
python tracking/test.py stark_lightning_X_trt baseline_rephead_4_lite_search5 --threads 0 --num_gpus 1
```
- 评估跟踪指标
```
python tracking/analysis_results_ITP.py --script stark_lightning_X_trt --config baseline_rephead_4_lite_search5
```
