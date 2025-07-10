
#include "main.hpp"


using namespace nvinfer1;

int main()
{
    std::string engine_name = "/home/zy/yolo11trt/best.wts.guo1zy";
    std::string img_dir = "/home/nyakori/桌面/yolov8/images/image_1.jpg";
    std::string labels_filename = "/home/zy/yolo11trt/predict/yolo/yolov8/model/classes.txt";
    int model_bboxes;
    Yolov8 yolo(engine_name, labels_filename, cv::Size(256, 256), 0.5, 0.5);
    cv::Mat test = cv::imread(img_dir);
    timeval tt1, tt2;
    YoloDetect result;
    test = cv::imread(img_dir);
    gettimeofday(&tt1, NULL);
    result = yolo.detect(test);
    gettimeofday(&tt2, NULL);
    double cloudtime = (tt2.tv_sec - tt1.tv_sec) * 1e3 + (tt2.tv_usec - tt1.tv_usec) / 1000.0;
    printf("all_timeuse %lf ms\n", cloudtime);
    yolo.drawResult(test, result);
    cv::imshow("sw", test);
    cv::waitKey(0);
}
