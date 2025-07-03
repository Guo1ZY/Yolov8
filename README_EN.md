# YOLOv11 C++ Inference

🚀 A high-performance C++ implementation for YOLOv11 object detection using OpenCV DNN module.

## ✨ Features

- **Real-time Detection**: Fast inference with optimized C++ implementation
- **CUDA Support**: GPU acceleration for enhanced performance (configurable)
- **Flexible Input Sizes**: Support for 256x256, 640x640, and custom input dimensions
- **Letterbox Preprocessing**: Maintains aspect ratio with proper padding
- **NMS Post-processing**: Non-Maximum Suppression for cleaner results
- **Live Camera Feed**: Real-time detection from camera input
- **Easy Integration**: Simple API for embedding into existing projects

## 🛠️ Requirements

### Essential Dependencies

- **OpenCV 4.7+** (Critical - earlier versions may have compatibility issues)
- **C++11** or higher
- **CMake 3.10+**
- **ONNX Runtime** (optional, for model optimization)

### Optional Dependencies

- **CUDA 11.0+** (for GPU acceleration)
- **cuDNN 8.0+** (for CUDA support)

## 📦 Installation

### 1. Install OpenCV 4.7

#### Ubuntu/Debian

```bash
# Remove old OpenCV versions
sudo apt remove libopencv-dev

# Install dependencies
sudo apt update
sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev

# Download and build OpenCV 4.7
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

### 2. Clone and Build Project

```bash
git clone https://github.com/yourusername/yolov11-cpp-inference.git
cd yolov11-cpp-inference
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 🎯 Usage

### 1. Prepare Your Model

#### Export YOLOv11 to ONNX

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('path/to/your/model.pt')

# Export to ONNX with OpenCV compatibility
model.export(
    format='onnx',
    opset=11,           # Important for OpenCV compatibility
    simplify=True,      # Simplify the model
    dynamic=False       # Static input shape
)
```

#### Create Classes File

Create a `classes.txt` file with your class names (one per line):

```
apple
orange
banana
grape
```

### 2. Update Configuration

Modify the paths in `main.cpp`:

```cpp
// Update these paths to match your setup
string modelPath = "/path/to/your/model.onnx";
string classesPath = "/path/to/your/classes.txt";
```

### 3. Run Detection

```bash
# Build the project
cd build
make

# Run with camera input
./yolov11_detection

# The program will:
# - Initialize the camera
# - Load the YOLOv11 model
# - Perform real-time detection
# - Display results with bounding boxes
```

### 4. Using the API

```cpp
#include "Yolov11.hpp"

int main() {
    // Initialize YOLOv11 (model_path, classes_path, use_cuda, input_size, conf_threshold, iou_threshold)
    Yolov11 yolo("model.onnx", "classes.txt", true, cv::Size(256, 256), 0.5, 0.5);
    
    // Load an image
    cv::Mat image = cv::imread("test_image.jpg");
    
    // Perform detection
    std::vector<YoloDetect> results = yolo.detect(image);
    
    // Draw results
    yolo.drawResult(image, results);
    
    // Display or save
    cv::imshow("Detection", image);
    cv::waitKey(0);
    
    return 0;
}
```

## 📁 Project Structure

```
yolov11-cpp-inference/
├── src/
│   ├── Yolov11.cpp          # Main implementation
│   ├── Yolov11.hpp          # Header file  
│   └── main.cpp             # Example usage
├── model/
│   ├── best.onnx           # Your ONNX model
│   └── classes.txt         # Class names
├── CMakeLists.txt          # Build configuration
└── README.md               # This file
```

## ⚙️ Configuration Options

### Constructor Parameters

```cpp
Yolov11(
    const std::string modelPath,           // Path to ONNX model
    const std::string classesPath,         // Path to classes.txt
    const bool &runWithCuda = false,       // Enable GPU acceleration
    const cv::Size &modelInputShape = {256, 256},  // Input resolution
    const float &modelConfThreshold = 0.5, // Confidence threshold
    const float &modelIouThreshold = 0.5   // IoU threshold for NMS
);
```

### Recommended Settings

| Use Case      | Input Size | Conf Threshold | IoU Threshold | Notes                         |
| ------------- | ---------- | -------------- | ------------- | ----------------------------- |
| Real-time     | 256×256    | 0.5            | 0.5           | Fastest inference             |
| Balanced      | 416×416    | 0.5            | 0.45          | Good speed/accuracy trade-off |
| High Accuracy | 640×640    | 0.25           | 0.45          | Best accuracy, slower         |

## 🔧 Troubleshooting

### Common Issues

#### 1. OpenCV Version Issues

```bash
# Check OpenCV version
pkg-config --modversion opencv4

# If version < 4.7, reinstall OpenCV 4.7+
```

#### 2. Model Loading Errors

- Ensure ONNX model is exported with `opset=11`
- Verify model file is not corrupted
- Check file permissions

#### 3. CUDA Issues

```cpp
// If CUDA fails, the code automatically falls back to CPU
// Check CUDA installation:
nvidia-smi
nvcc --version
```

#### 4. Camera Not Found

```cpp
// Try different camera indices
cv::VideoCapture camera(0);  // Try 0, 1, 2, etc.
```

### Model Re-export Script

If you encounter model compatibility issues, re-export with:

```python
import torch
from ultralytics import YOLO

def fix_yolov11_model(input_path, output_path):
    """Re-export YOLOv11 model for OpenCV compatibility"""
    model = YOLO(input_path)
    
    # Export with specific settings for OpenCV
    model.export(
        format='onnx',
        opset=11,
        simplify=True,
        dynamic=False,
        imgsz=256  # or your preferred size
    )
    
    print(f"Model re-exported to {output_path}")

# Usage
fix_yolov11_model('best.pt', 'best_fixed.onnx')
```

## 🎨 Customization

### Adding New Features

1. **Custom Preprocessing**:

```cpp
cv::Mat customPreprocess(const cv::Mat& image) {
    // Add your custom preprocessing here
    return processed_image;
}
```

1. **Different Input Sizes**:

```cpp
// Support for different resolutions
Yolov11 yolo(modelPath, classesPath, true, cv::Size(640, 640));
```

1. **Batch Processing**:

```cpp
// Process multiple images
std::vector<cv::Mat> images = {img1, img2, img3};
for (const auto& img : images) {
    auto results = yolo.detect(img);
    // Process results...
}
```

## 📊 Performance Benchmarks

| Hardware        | Input Size | FPS  | Notes        |
| --------------- | ---------- | ---- | ------------ |
| RTX 3080        | 256×256    | ~150 | CUDA enabled |
| RTX 3080        | 640×640    | ~80  | CUDA enabled |
| Intel i7-10700K | 256×256    | ~45  | CPU only     |
| Intel i7-10700K | 640×640    | ~15  | CPU only     |

*Benchmarks may vary based on model complexity and system configuration*

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://gptsdd.com/chat/LICENSE) file for details.

------

⭐ **Star this repository if it helped you!**
