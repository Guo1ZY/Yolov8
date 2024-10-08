# README

------

Environment requirement

- ubuntu20.04
- python>=3.8
- cuda=11.8
- GPU（my computer is RTX 3050 Laptop 4G)
- opencv>=4.7(4.7 recommended)
- onnxruntime

------

本代码为Yolov8 C++ 推理onnx的代码--分支ONNX_CPP

参考ultralytics-yolov8_onnx_CPP_example

在main.cpp中修改模型以及classes.txt路径

模型后缀为.onnx

类别根据自己训练所分的类别确定，如果和模型对不上会报错



