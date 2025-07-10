file(GLOB_RECURSE main_srcs CONFIGURE_DEPENDS include/*.hpp sources/main.cpp)
add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
add_executable(run ${main_srcs})

  # cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
  
    # tensorrt
include_directories(/workspace/shared/TensorRT-8.6.1.6/include)
link_directories(/workspace/shared/TensorRT-8.6.1.6/lib)
  
  
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

target_include_directories(run PUBLIC include)
target_link_libraries(run PUBLIC yolo)