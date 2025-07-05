/**
 * @file main.cpp
 * @author zzzing (1226196281@qq.com)
 * @brief yolov8的cpp和python混合编程代码
 * @version 0.1
 * @date 2023-10-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "main.hpp"

#include <opencv2/opencv.hpp>

#include <vector>

using namespace std;
using namespace cv;

/*训练好的模型路径*/
// string modelPath = "/home/zy/下载/yolov8n-seg.pt";
string modelPath = "/home/zy/桌面/yolo_track/Cpp_Python_Samples/py/yolov8/model/ball/best.pt";

/*分类标签路径*/
string classesPath = "/home/zy/桌面/yolo_track/Cpp_Python_Samples/py/yolov8/model/ball/classes.txt";

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open USB camera" << std::endl;
        return -1;
    }
    cv::Mat image;
    /*初始化Python*/
    Python_Initialize();

    /*创建一个yolo class*/
    Yolov8 yolo(modelPath, classesPath);

    // cv::VideoCapture cap(0);
    // if (!cap.isOpened())
    // {
    //     std::cerr << "Failed to open USB camera" << std::endl;
    //     return -1;
    // }
    // cv::Mat image;
    while (1){
    cap >> image;
    /*检测*/
    std::vector<YoloDetect> result = yolo.detect(image);
    /*绘图*/
    yolo.drawResult(image, result);
    if(result.size() !=0){
        std::cout << "检测到球" << std::endl;
    
    //输出检测结果
    for (int i = 0; i < result.size(); i++)
    {
        //id
        std::cout << "id: " << result[i].id << std::endl;
        //置信度
        std::cout << "confidence: " << result[i].confidence << std::endl;
    }
    }
    else
    {
        std::cout << "未检测到球" << std::endl;
    }
    imshow("image", image);
    waitKey(10);
    
    }
    return 0;

}