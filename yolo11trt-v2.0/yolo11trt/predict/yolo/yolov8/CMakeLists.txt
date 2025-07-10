cmake_minimum_required(VERSION 3.10)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# 设置构建类型
set(CMAKE_BUILD_TYPE Debug)

# 指定 CUDA 编译器
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# 启用 CUDA 语言支持
enable_language(CUDA)

# Include directories
include_directories(
    /usr/local/cuda/include
    /workspace/shared/TensorRT-8.6.1.6/include
)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 源文件（根据实际文件名调整）
set(YOLOV8_SRC sources/Yolov8.cu)  # 请确保这里的路径和文件名正确
set(YOLOV8_HEADER include/Yolov8.hpp)  # 头文件路径

# 添加库
add_library(yolov8 STATIC ${YOLOV8_SRC} ${YOLOV8_HEADER})

# 链接库
target_link_libraries(yolov8 nvinfer cudart ${OpenCV_LIBS})

# 指定公共头文件目录
target_include_directories(yolov8 PUBLIC include)
