file(GLOB_RECURSE main_srcs CONFIGURE_DEPENDS include/*.hpp sources/main.cpp)

add_executable(run ${main_srcs})

add_thread(run)
# add_python(run)
add_opencv(run)


target_include_directories(run PUBLIC include)
target_link_libraries(run PUBLIC yolo)
# target_link_libraries(run PUBLIC ZedLibrary)