#ifndef __MAIN_HPP
#define __MAIN_HPP

#include <iostream>

// #include <thread>

#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include "Yolov8.hpp"

#define ROS_INFO_GREEN(msg) ROS_INFO("\033[0;32m%s\033[0m", msg)
#define ROS_INFO_RED(msg) ROS_INFO("\033[0;31m%s\033[0m", msg)

#endif
