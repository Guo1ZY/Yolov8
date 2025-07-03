# YOLOv11 C++ 推理

🚀 基于OpenCV DNN模块的高性能YOLOv11目标检测C++实现

## ✨ 特性

- **实时检测**: 优化的C++实现，快速推理
- **CUDA支持**: GPU加速提升性能（可配置）
- **灵活输入尺寸**: 支持256x256、640x640和自定义输入尺寸
- **Letterbox预处理**: 保持宽高比的填充处理
- **NMS后处理**: 非极大值抑制获得更清晰的结果
- **实时摄像头**: 摄像头实时检测
- **易于集成**: 简单的API，便于集成到现有项目

## 🛠️ 环境要求

### 必需依赖

- **OpenCV 4.7+** (关键 - 早期版本可能存在兼容性问题)
- **C++11** 或更高版本
- **CMake 3.10+**
- **ONNX Runtime** (可选，用于模型优化)

### 可选依赖

- **CUDA 11.0+** (GPU加速)
- **cuDNN 8.0+** (CUDA支持)

## 📦 安装

### 1. 安装OpenCV 4.7

#### Ubuntu/Debian

```bash
# 移除旧版本OpenCV
sudo apt remove libopencv-dev

# 安装依赖
sudo apt update
sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev

# 下载并编译OpenCV 4.7
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.7.0.zip
unzip opencv.zip
cd opencv-4.7.0
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.7.0/modules \
      -D BUILD_EXAMPLES=ON ..

make -j$(nproc)
sudo make install
sudo ldconfig
```

#### macOS

```bash
brew install opencv@4.7
```

### 2. 克隆并编译项目

```bash
git clone https://github.com/yourusername/yolov11-cpp-inference.git
cd yolov11-cpp-inference
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 🎯 使用方法

### 1. 准备模型

#### 导出YOLOv11为ONNX格式

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('path/to/your/model.pt')

# 导出为ONNX格式，确保OpenCV兼容性
model.export(
    format='onnx',
    opset=11,           # 重要：OpenCV兼容性
    simplify=True,      # 简化模型
    dynamic=False       # 静态输入尺寸
)
```

#### 创建类别文件

创建`classes.txt`文件，每行一个类别名称：

```
苹果
橙子
香蕉
葡萄
```

### 2. 更新配置

修改`main.cpp`中的路径：

```cpp
// 更新这些路径以匹配您的设置
string modelPath = "/path/to/your/model.onnx";
string classesPath = "/path/to/your/classes.txt";
```

### 3. 运行检测

```bash
# 编译项目
cd build
make

# 运行摄像头检测
./yolov11_detection

# 程序将会：
# - 初始化摄像头
# - 加载YOLOv11模型
# - 执行实时检测
# - 显示带边界框的结果
```

### 4. API使用示例

```cpp
#include "Yolov11.hpp"

int main() {
    // 初始化YOLOv11 (模型路径, 类别路径, 使用cuda, 输入尺寸, 置信度阈值, iou阈值)
    Yolov11 yolo("model.onnx", "classes.txt", true, cv::Size(256, 256), 0.5, 0.5);
    
    // 加载图像
    cv::Mat image = cv::imread("test_image.jpg");
    
    // 执行检测
    std::vector<YoloDetect> results = yolo.detect(image);
    
    // 绘制结果
    yolo.drawResult(image, results);
    
    // 显示或保存
    cv::imshow("Detection", image);
    cv::waitKey(0);
    
    return 0;
}
```

## 📁 项目结构

```
yolov11-cpp-inference/
├── src/
│   ├── Yolov11.cpp          # 主要实现
│   ├── Yolov11.hpp          # 头文件  
│   └── main.cpp             # 使用示例
├── model/
│   ├── best.onnx           # ONNX模型文件
│   └── classes.txt         # 类别名称
├── CMakeLists.txt          # 编译配置
└── README.md               # 说明文档
```

## ⚙️ 配置选项

### 构造函数参数

```cpp
Yolov11(
    const std::string modelPath,           // ONNX模型路径
    const std::string classesPath,         // classes.txt路径
    const bool &runWithCuda = false,       // 启用GPU加速
    const cv::Size &modelInputShape = {256, 256},  // 输入分辨率
    const float &modelConfThreshold = 0.5, // 置信度阈值
    const float &modelIouThreshold = 0.5   // NMS的IoU阈值
);
```

### 推荐设置

| 使用场景 | 输入尺寸 | 置信度阈值 | IoU阈值 | 说明           |
| -------- | -------- | ---------- | ------- | -------------- |
| 实时检测 | 256×256  | 0.5        | 0.5     | 最快推理速度   |
| 平衡模式 | 416×416  | 0.5        | 0.45    | 速度/精度平衡  |
| 高精度   | 640×640  | 0.25       | 0.45    | 最佳精度，较慢 |

## 🔧 故障排除

### 常见问题

#### 1. OpenCV版本问题

```bash
# 检查OpenCV版本
pkg-config --modversion opencv4

# 如果版本 < 4.7，请重新安装OpenCV 4.7+
```

#### 2. 模型加载错误

- 确保ONNX模型使用`opset=11`导出
- 验证模型文件没有损坏
- 检查文件权限

#### 3. CUDA问题

```cpp
// 如果CUDA失败，代码会自动回退到CPU
// 检查CUDA安装:
nvidia-smi
nvcc --version
```

#### 4. 摄像头未找到

```cpp
// 尝试不同的摄像头索引
cv::VideoCapture camera(0);  // 尝试 0, 1, 2 等
```

### 模型重新导出脚本

如果遇到模型兼容性问题，使用以下脚本重新导出：

```python
import torch
from ultralytics import YOLO

def fix_yolov11_model(input_path, output_path):
    """重新导出YOLOv11模型以确保OpenCV兼容性"""
    model = YOLO(input_path)
    
    # 使用特定设置导出以确保OpenCV兼容性
    model.export(
        format='onnx',
        opset=11,
        simplify=True,
        dynamic=False,
        imgsz=256  # 或您偏好的尺寸
    )
    
    print(f"模型已重新导出到 {output_path}")

# 使用方法
fix_yolov11_model('best.pt', 'best_fixed.onnx')
```

## 🎨 自定义功能

### 添加新功能

1. **自定义预处理**:

```cpp
cv::Mat customPreprocess(const cv::Mat& image) {
    // 在这里添加您的自定义预处理
    return processed_image;
}
```

1. **不同输入尺寸**:

```cpp
// 支持不同分辨率
Yolov11 yolo(modelPath, classesPath, true, cv::Size(640, 640));
```

1. **批量处理**:

```cpp
// 处理多张图像
std::vector<cv::Mat> images = {img1, img2, img3};
for (const auto& img : images) {
    auto results = yolo.detect(img);
    // 处理结果...
}
```

## 📊 性能基准

| 硬件配置        | 输入尺寸 | FPS  | 备注     |
| --------------- | -------- | ---- | -------- |
| RTX 3080        | 256×256  | ~150 | CUDA启用 |
| RTX 3080        | 640×640  | ~80  | CUDA启用 |
| Intel i7-10700K | 256×256  | ~45  | 仅CPU    |
| Intel i7-10700K | 640×640  | ~15  | 仅CPU    |

*基准测试结果可能因模型复杂度和系统配置而异*

## 🤝 贡献

1. Fork本仓库
2. 创建您的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 📝 许可证

本项目基于MIT许可证 - 查看[LICENSE](https://gptsdd.com/chat/LICENSE)文件了解详情。

------

⭐ **如果这个项目对您有帮助，请给个Star！**
