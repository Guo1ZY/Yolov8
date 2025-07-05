/*
 * @Author: Guo1ZY /home/zy/catkin_ws/src/Kinect_R2/build/devel/lib/kinect_r2/kinect_r2@github.com
 * @Date: 2023-11-21 16:30:50
 * @LastEditors: Guo1ZY /home/zy/catkin_ws/src/Kinect_R2/build/devel/lib/kinect_r2/kinect_r2@github.com
 * @LastEditTime: 2024-03-23 19:41:37
 * @FilePath: /Cpp_Onnx_Samples/main/sources/main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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

using namespace std;
using namespace cv;

/*训练好的模型路径*/
string modelPath = "/home/zy/ws_livox/src/apple_detection/yolo/yolov8/model/best.onnx";

/*分类标签路径*/
string classesPath = "/home/zy/ws_livox/src/apple_detection/yolo/yolov8/model/classes.txt";

// int main()
// {
//     Yolov8 yolo(modelPath, classesPath, true, cv::Size(256, 256), 0.5, 0.5);

//     //打开相机
    
//     // std::vector<std::string> imageNames;
//     // glob("/home/zy/桌面/yolov8_train/R2_谷仓/谷仓训练3.19/data/images/*.jpg", imageNames);

//     for (size_t i = 0; i < imageNames.size(); i++)
//     {
//         cv::Mat frame = cv::imread(imageNames[i]);

//         timeval tt1, tt2;
//         gettimeofday(&tt1, NULL);

//         // Inference starts here...
//         std::vector<YoloDetect> results = yolo.detect(frame);

//         gettimeofday(&tt2, NULL);
//         float diff_time = 1000 * (tt2.tv_sec - tt1.tv_sec) + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
//         cout << "diff time " << diff_time << endl;

//         std::cout << "Number of detections:" << results.size() << std::endl;

//         yolo.drawResult(frame, results);

//         cv::imshow("image", frame);
//         cv::waitKey(0);
//     }

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

//     return 0;
// }

int main()
{
    // Ctrl C 中断
    // signal(SIGINT, ctrlc);

    Yolov8 yolo(modelPath, classesPath, true, cv::Size(256, 256), 0.5, 0.5);
    cv::VideoCapture camera(2);
    if (!camera.isOpened())
    {
        cerr << "Error: Camera not found." << endl;
        return -1;
    }

    while (1)
    {
        // if (ctrl_c_pressed == true)
        //     break;

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