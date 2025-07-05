macro(add_yaml target)
    find_package(yaml-cpp REQUIRED)
    target_link_libraries(${target} PRIVATE yaml-cpp)
endmacro()

macro(add_thread target)
    find_package(Threads REQUIRED)
    target_link_libraries(${target} PRIVATE Threads::Threads)
endmacro()

macro(add_boost target)
    find_package(Boost COMPONENTS thread system REQUIRED)
    target_link_libraries(${target} PUBLIC Boost::thread Boost::system)
endmacro()

macro(add_opencv target)
    find_package(OpenCV 4.7 REQUIRED)
    target_link_libraries(${target} PUBLIC ${OpenCV_LIBS})
endmacro()

macro(add_python target)
    # set python path
    set(PYTHON_INCLUDE_DIRS "/home/zzzing/anaconda3/envs/pytorch/include/python3.11")
    target_include_directories(${target} PUBLIC ${PYTHON_INCLUDE_DIRS})
    
    target_link_directories(${target} PUBLIC "/home/zzzing/anaconda3/envs/pytorch/lib/python3.11/config-3.11-x86_64-linux-gnu")
    set(PYTHON_LIBRARIES "/home/zzzing/anaconda3/envs/pytorch/lib/libpython3.11.so")

    # get numpy include
    execute_process(
        COMMAND python3 -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    target_include_directories(${target} PUBLIC ${NumPy_INCLUDE_DIRS})

    target_link_libraries(${target} PUBLIC ${PYTHON_LIBRARIES})
endmacro()

macro(add_ros target)
    find_package(catkin REQUIRED COMPONENTS
        cv_bridge
        image_transport
        roscpp
        sensor_msgs
        std_msgs
    )

    target_include_directories(${target} PUBLIC ${catkin_INCLUDE_DIRS})
    target_link_libraries(${target} PUBLIC ${catkin_LIBRARIES})
endmacro(add_ros)
