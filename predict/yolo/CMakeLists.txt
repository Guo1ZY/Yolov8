add_library(yolo yolo.h)

add_subdirectory(yolov8)

target_link_libraries(yolo PUBLIC yolov8)