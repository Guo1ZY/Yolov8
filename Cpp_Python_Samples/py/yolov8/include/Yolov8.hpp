#ifndef __YOLOV8_HPP
#define __YOLOV8_HPP

#include <iostream>

#include <vector>

#include <opencv2/opencv.hpp>

#include <Python.h>

/**
 * @brief 初始化Python环境，如果初始化失败，则会调用std::exit(1)退出程序
 *
 * @return 是否初始化成功
 */
int Python_Initialize();

/**
 * @brief Yolov8检测结果
 *
 */
struct YoloDetect
{
    int class_index = 0;                    // 类别索引
    std::string class_name = "";            // 类别名
    float confidence = 0.0;                 // 置信度
    cv::Rect box = cv::Rect(0, 0, 0, 0);    // 矩形
    cv::Scalar color = cv::Scalar(0, 0, 0); // 颜色
};

/**
 * @brief 选择Yolo推理模式
 *
 */
enum class YoloMode
{
    Pytorch = 1,
    Onnx = 2
};

/**
 * @brief Yolov8模型
 *
 */
class Yolov8
{
private:
    /*为Python添加的import路径*/
    std::string pySysPath = "../py/yolov8/py";
    /*置信度阈值*/
    float confThreshold = 0.5;
    /*是否使用cuda推理*/
    bool cudaEnable = true;
    /*选择Yolo推理模式*/
    YoloMode yolomode = YoloMode::Pytorch;

public:
    /**
     * @brief 构造函数，同时初始化Yolov8模型和分类名
     *
     * @param modelPath Yolov8模型路径
     * @param classesPath 分类名字路径
     */
    Yolov8(const std::string modelPath, const std::string classesPath);

    /**
     * @brief Yolov8检测
     *
     * @param img 待检测图片
     * @return std::vector<YoloDetect> 检测结果
     */
    std::vector<YoloDetect>
    detect(const cv::Mat &image);

    /**
     * @brief 绘制检测结果
     *
     * @param image 待绘制图片
     * @param result 检测结果
     */
    void drawResult(cv::Mat &image, const std::vector<YoloDetect> &results);

private:
    /*import导入的模块*/
    PyObject *pyModule;

    /*提取Python Yolov8类对象*/
    PyObject *pyYolov8Class;

    /*实例化Python Yolov8类*/
    PyObject *pyYolov8Instance;

    /*提取Python Yolov8类的检测函数*/
    PyObject *pyYolov8Detect;

    /*分类的名字*/
    std::vector<std::string>
        classes;

    /*随机数生成器*/
    cv::RNG rng;

    /**
     * @brief 将Mat图片转为python array
     *
     * @param image Mat图片
     * @return PyObject* python array
     */
    PyObject *ConvertImageTopyArray(const cv::Mat &image);

    /**
     * @brief 加载Yolov8模型
     *
     * @param modelPath Yolov8模型路径
     */
    void loadYolov8Model(const std::string modelPath);

    /**
     * @brief 加载分类名字
     *
     * @param classesPath 分类名字路径
     */
    void loadClassesNames(const std::string classesPath);

    /**
     * @brief 生成随机颜色
     *
     * @return cv::Scalar 随机颜色
     */
    cv::Scalar generateRandomColor();

private:
/****************打印颜色定义****************/
#define COUT_RED_START std::cout << "\033[1;31m";
#define COUT_GREEN_START std::cout << "\033[1;32m";
#define COUT_YELLOW_START std::cout << "\033[1;33m";
#define COUT_BLUE_START std::cout << "\033[1;34m";
#define COUT_PURPLE_START std::cout << "\033[1;35m";
#define COUT_CYAN_START std::cout << "\033[1;36m";
#define COUT_WHITE_START std::cout << "\033[1;37m";
#define COUT_COLOR_END std::cout << "\033[0m";
};

#endif