#ifndef __YOLOV11_HPP
#define __YOLOV11_HPP
// *version 3.0
// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/**
 * @brief Yolov11检测结果
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
 * @brief Yolov11模型
 *
 */
class Yolov11
{
public:
    /**
     * @brief 构造函数，同时初始化Yolov11模型和分类名
     *
     * @param modelPath Yolov11模型路径
     * @param classesPath 分类名字路径
     * @param runWithCuda 是否使用cuda
     * @param modelInputShape 模型输入尺寸
     * @param modelConfThreshold 置信度阈值
     * @param modelIouThreshold iou阈值
     */
    Yolov11(const std::string modelPath, const std::string classesPath, const bool &runWithCuda = false, const cv::Size &modelInputShape = {256, 256}, const float &modelConfThreshold = 0.5, const float &modelIouThreshold = 0.5);

    /**
     * @brief Yolov11检测
     *
     * @param img 待检测图片
     * @return std::vector<YoloDetect> 检测结果
     */
    std::vector<YoloDetect> detect(const cv::Mat &image);

    /**
     * @brief 绘制检测结果
     *
     * @param image 待绘制图片
     * @param result 检测结果
     */
    void drawResult(cv::Mat &image, const std::vector<YoloDetect> &results);

private:
    /*Yolov11网络*/
    cv::dnn::Net net;

    /*输入网络的尺寸*/
    cv::Size inputShape;
    /*置信度阈值*/
    float confThreshold;
    /*iou阈值*/
    float iouThreshold;

    /*分类的名字*/
    std::vector<std::string> classes;

    /*随机数生成器*/
    cv::RNG rng;

    /**
     * @brief 加载Yolov11模型
     *
     * @param modelPath Yolov11模型路径
     * @param runWithCuda 是否使用cuda
     */
    void loadYolov11Model(const std::string &modelPath, const bool &runWithCuda = true);

    /**
     * @brief 加载分类名字
     *
     * @param classesPath 分类名字路径
     */
    void loadClassesNames(const std::string classesPath);

    /**
     * @brief 处理网络输出层数据
     *
     * @param outputs 网络输出层数据
     * @param image_cols 输入图片的宽
     * @param image_rows 输入图片的高
     * @return std::vector<YoloDetect> 处理后的检测结果
     */
    std::vector<YoloDetect> postprocess(std::vector<cv::Mat> &outputs, const int image_cols, const int image_rows);

    /**
     * @brief 将图片转换为正方形
     *
     * @param source 待转换的图片
     * @return cv::Mat 正方形图片
     */
    cv::Mat formatToSquare(const cv::Mat &source);

    /**
     * @brief 生成随机颜色
     *
     * @return cv::Scalar 随机颜色
     */
    cv::Scalar generateRandomColor();

    /**
     * @brief 图像预处理 - 包含letterbox填充
     *
     * @param image 输入图像
     * @param newShape 目标尺寸
     * @return cv::Mat 预处理后的图像
     */
    cv::Mat letterbox(const cv::Mat &image, const cv::Size &newShape = cv::Size(256, 256));

    /**
     * @brief 测试推理以验证模型加载
     */
    void testInference();

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