# 介绍

最终测试效果：3.1ms，包括图片传入，即yolo.predict(image)的时间，测试平台13600kf+4070ti，yolov8m.pt，输入图像512*512）



修改来自源码：

[]: https://github.com/wang-xinyu/tensorrtx



1、api相较与cpp_onnx_sample没有任何改变，文件结构基本没有改变

2、优化了光敏性癫痫患者的使用体验





## 注意事项

1、必须是20系以上的nvidia游戏显卡，或者计算卡（应该不会有人有计算卡吧wwww）

2、训练出来的模型不能跨不同型号的显卡运行，不能跨cuda，cudnn，tensorRT版本运行



# tensorRT的安装

1、推荐版本8.6.1.6，原因是这是开源代码的推荐版本，在安装之前，你需要提前安装好cuda11.8和cudnn8.6（python版的tensor装不装tensorRT随意，但要有torch）

2、下载链接

[]: https://developer.nvidia.com/nvidia-tensorrt-8x-download

找到[](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz)

下载下来解压到你喜欢的文件夹，可以像这样/usr/local/TensorRT-8.6.1.6

3、你还需要把tensorRT添加到环境变量中

```
vim ~/.bashrc
```

在文件末尾添加（注意路径）

```
export LD_LIBRARY_PATH=/usr/local/TensorRT-8.5.3.1/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda/bin:/usr/local/TensorRT-8.6.1.6/bin
```

保存并推出

```
source ~/.bashrc
```

检查结果

```
echo $PATH
```

# 获取tensorRT模型

## 获取onnx模型

首先你要先弄出yolo的.pt模型，请前往Yolov8-master获取

## 获取tensorRT支持的模型

模型后缀可以瞎写，我试过了，甚至支持中文和乱七八糟的标点符号

1、进入get_model

2、执行gen_wts.py将.pt转化为wts（记得改路径），执行时pip安装它提示你缺失的库

3、修改include中的config.h,\#define你想使用的精度（FP32是原精度），kNumClass改为你预测的种类数，kInputH和kInputW是你训练时选择的处理图片的长宽

3、修改main.cpp，wts_name为你wts模型路径，engine_name是你保存出的模型名字，type是你最初用来训练的yolo模型后缀，比如yolov8n对应n,其它的自行类比

4、运行main.cpp，若编译报错请检查cmakelists，保存出的模型会在build目录下

# 利用tensorRT模型推理

可以参考predict的示例代码











