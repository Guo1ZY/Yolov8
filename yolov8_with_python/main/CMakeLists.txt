file(GLOB_RECURSE main_srcs CONFIGURE_DEPENDS include/*.hpp sources/main.cpp)

add_executable(run ${main_srcs})

# add_thread(run)
add_python(run)
add_opencv(run)
add_ros(run)

target_include_directories(run PUBLIC include)
target_link_libraries(run PUBLIC py)

file(GLOB_RECURSE yolo_detect CONFIGURE_DEPENDS include/*.hpp sources/yolo_detect.cpp)

add_executable(yolo_detect ${yolo_detect})

add_python(yolo_detect)
add_opencv(yolo_detect)
add_ros(yolo_detect)

target_include_directories(yolo_detect PUBLIC include)
target_link_libraries(yolo_detect PUBLIC py)

file(GLOB_RECURSE test_camera CONFIGURE_DEPENDS include/*.hpp sources/test.cpp)

add_executable(test_camera ${test_camera})

add_python(test_camera)
add_opencv(test_camera)
add_ros(test_camera)

target_include_directories(test_camera PUBLIC include)
target_link_libraries(test_camera PUBLIC py)