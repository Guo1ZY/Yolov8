# set the minimum version request for cmake
cmake_minimum_required(VERSION 3.16)

# set the LANGUAGE
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # 建议关闭 C++ 扩展
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# set the project
project(Uart_V2
LANGUAGES C CXX CUDA  # 添加 CUDA
VERSION 2.0.1         # set the version
)

# set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# add the cmake submodule for convenience
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")

# be careful about the option
set(CMAKE_BUILD_TYPE Debug)

# add subdirectory
add_subdirectory(yolo)
add_subdirectory(main)

