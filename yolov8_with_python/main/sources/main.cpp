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
string modelPath = "../py/yolov8/model/silver/best.pt";

/*分类标签路径*/
string classesPath = "../py/yolov8/model/silver/classes.txt";

int main()
{
    // /*初始化Python*/
    // Python_Initialize();

    // /*创建一个yolo class*/
    // Yolov8 yolo(modelPath, classesPath);

    // Mat image = imread("/home/zzzing/桌面/3_22_gongye/jpgs/3_1.jpg");

    // /*检测*/
    // vector<YoloDetect> result = yolo.detect(image);
    // /*绘图*/
    // yolo.drawResult(image, result);

    // imshow("image", image);
    // waitKey(0);

    // image = imread("/home/zzzing/桌面/3_22_gongye/jpgs/112_1.jpg");

    // timeval tt1, tt2;
    // gettimeofday(&tt1, NULL);
    // result = yolo.detect(image);
    // gettimeofday(&tt2, NULL);

    // /*运行时间*/
    // float diff_time = 1e3 * (tt2.tv_sec - tt1.tv_sec) + (tt2.tv_usec - tt1.tv_usec) / 1000.0f;
    // cout << "diff time " << diff_time << endl;
    // yolo.drawResult(image, result);

    // imshow("image", image);
    // waitKey(0);

    return 0;
}