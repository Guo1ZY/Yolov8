/*
 * @Author: hejia
 * @Date: 2024-11-02 22:11:33
 * @LastEditors: hejia
 * @LastEditTime: 2024-11-03 10:17:42
 * @Description: 
 * @FilePath: /predict/main/sources/main.cpp
 */
#include "main.hpp"

using namespace std;
using namespace cv;

/*训练好的模型路径*/
string modelPath = "/home/zy/yolo11trt/best.wts.guo1zy";

/*分类标签路径*/
string classesPath = "../yolo/yolov8/model/classes.txt";

int main()
{
    Yolov8 yolo(modelPath, classesPath, true, cv::Size(256, 256), 0.5, 0.5);

   cv::VideoCapture camera(2); // 打开摄像头设备2
    if (!camera.isOpened()) {
        std::cerr << "Failed to open camera." << std::endl;
        return -1;
    }

    // 设置摄像头分辨率为 640x480
        camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;
    while (true) {
        camera >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to capture frame." << std::endl;
            break;
        }

        timeval tt1, tt2;
        gettimeofday(&tt1, NULL);

        std::vector<YoloDetect> results = yolo.detect(frame);

        gettimeofday(&tt2, NULL);
        float diff_time = 1000 * (tt2.tv_sec - tt1.tv_sec) + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
        cout << "diff time " << diff_time << endl;

        std::cout << "Number of detections:" << results.size() << std::endl;

        yolo.drawResult(frame, results);

        cv::imshow("image", frame);
        if (cv::waitKey(1) == 27) { // 按下 ESC 键退出
            break;
        }
    }

    return 0;
}