/**
 * @file main.cpp
 * @author Guo1ZY
 * @brief yolov11的cpp和python混合编程代码
 * @version 0.2
 * @date 2024-12-14
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "main.hpp"
#include "Yolov11.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <sys/time.h>
using namespace std;
using namespace cv;

/*训练好的模型路径*/
string modelPath = "/home/zy/Yolov11_Train/output_v11/weights/best.onnx";

/*分类标签路径*/
string classesPath = "/home/zy/ws_livox/src/apple_detection/yolo/yolov11/model/classes.txt";

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
    timeval t1, t2;

    // 初始化YOLOv11模型，使用256x256输入尺寸
    Yolov11 yolo(modelPath, classesPath, true, cv::Size(256, 256), 0.5, 0.5);

    cv::VideoCapture camera(2);
    if (!camera.isOpened())
    {
        cerr << "Error: Camera not found." << endl;
        return -1;
    }

    // 设置摄像头分辨率
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    while (1)
    {
        if (ctrl_c_pressed == true)
            break;

        cv::Mat image;
        camera >> image;

        if (image.empty())
        {
            cerr << "Error: Empty frame captured" << endl;
            continue;
        }
        gettimeofday(&t1, NULL);
        std::vector<YoloDetect> result = yolo.detect(image);

        /*绘图*/
        yolo.drawResult(image, result);

        // 显示检测结果数量
        string info = "Detections: " + to_string(result.size());
        cv::putText(image, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        imshow("YOLOv11 Detection", image);
        gettimeofday(&t2, NULL);

        // 计算时间ms
        double timeuse = 1000 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("Processing time: %.2f ms, FPS: %.1f\n", timeuse, 1000.0 / timeuse);

        char key = waitKey(1);
        if (key == 27 || key == 'q') // ESC or 'q' to exit
            break;
    }

    camera.release();
    cv::destroyAllWindows();

    return 0;
}