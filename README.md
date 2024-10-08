# README

本代码为C++调用Python模块进行Yolov8目标检测示例代码

## 需要注意的路径

1. `cmake/add_dependency.cmake`中的 `add_python`，请根据本地Python环境修改为对应路径
2. `Yolov8.hpp`中，为Python添加的import路径，当前为cmake相对路径
3. `main.cpp`中，加载的训练好的pytorch模型和分类标签路径
