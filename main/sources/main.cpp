/**
 * @file main.cpp
 * @author Guo1ZY
 * @brief yolov8的cpp和python混合编程代码
 * @version 0.1
 * @date 2023-10-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "main.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;


/*训练好的模型路径*/
string modelPath = "../best.onnx";

/*分类标签路径*/
string classesPath = "../classes.txt";

// ctrl c中断
#include <signal.h>
bool ctrl_c_pressed = false;
void ctrlc(int)
{
    ctrl_c_pressed = true;
}

int main()
{
    // Ctrl C 中断
    signal(SIGINT, ctrlc);

    Yolov8 yolo(modelPath, classesPath, true, cv::Size(256, 256), 0.5, 0.5);
    cv::VideoCapture camera(0);
    if (!camera.isOpened())
    {
        cerr << "Error: Camera not found." << endl;
        return -1;
    }

    while (1)
    {
        if (ctrl_c_pressed == true)
            break;

        cv::Mat image;
        camera >> image;
        std::vector<YoloDetect> result = yolo.detect(image);
        /*绘图*/
        yolo.drawResult(image, result);

        imshow("img", image);

        waitKey(10);
    }
    return 0;
}
