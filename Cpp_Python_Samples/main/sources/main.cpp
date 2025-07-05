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
string modelPath = "/home/zy/Yolov11_Train/output/weights/best.pt";

/*分类标签路径*/
string classesPath = "/home/zy/ws_livox/src/apple_detection/yolo/yolov8/model/classes.txt";

int main()
{
    /*初始化Python*/
    Python_Initialize();

    /*创建一个yolo class*/
    Yolov8 yolo(modelPath, classesPath);
    timeval tt1, tt2;
    // Mat image = imread("/home/zzzing/桌面/3_22_gongye/jpgs/3_1.jpg");
    cv::VideoCapture camera(2);
    if (!camera.isOpened())
    {
        cerr << "Error: Camera not found." << endl;
        return -1;
    }
    /*检测*/
    while (1)
    {
        // if (ctrl_c_pressed == true)
        //     break;

        cv::Mat image;
        camera >> image;
        gettimeofday(&tt1, NULL);
        printf("___________________________");
        std::vector<YoloDetect> result = yolo.detect(image);
        /*绘图*/
        yolo.drawResult(image, result);
        gettimeofday(&tt2, NULL);
        float diff_time = 1e3 * (tt2.tv_sec - tt1.tv_sec) + (tt2.tv_usec - tt1.tv_usec) / 1000.0f;
        cout << "diff time " << diff_time << endl;
        imshow("img", image);

        waitKey(10);
    }
    return 0;
}