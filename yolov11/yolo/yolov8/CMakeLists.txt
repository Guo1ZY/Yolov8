file(GLOB_RECURSE yolov8_srcs CONFIGURE_DEPENDS include/*.hpp sources/*.cpp)

add_library(yolov8 STATIC ${yolov8_srcs})

add_opencv(yolov8)
# add_python(yolov8)

target_include_directories(yolov8 PUBLIC include)