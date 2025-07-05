add_library(py py.h)

add_subdirectory(yolov8)

target_link_libraries(py PUBLIC yolov8)