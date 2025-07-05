#include "main.hpp"

std::string modelPath = "/home/zzzing/桌面/catkin_ws/src/yolov8_with_python/py/yolov8/model/ball/best.pt";
std::string classesPath = "/home/zzzing/桌面/catkin_ws/src/yolov8_with_python/py/yolov8/model/ball/classes.txt";

Yolov8 *yolo = nullptr;

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    static uint8_t init_count = 0;

    if (init_count <= 10)
    {
        init_count++;
        return;
    }

    // 将图像转换openCV的格式
    cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image.clone();

    if (image.empty())
    {
        ROS_INFO_RED("Get An Empty Image.");
        return;
    }

    cv::imshow("view", image); // 输出到窗口
    cv::waitKey(0);

    // std::vector<YoloDetect> result = yolo->detect(image); // 检测图片

    // yolo->drawResult(image, result); // 绘制检测结果

    cv::imshow("view", image); // 输出到窗口
    cv::waitKey(1);            // 一定要有wiatKey(),要不然是黑框或者无窗口
}

int main(int argc, char **argv)
{
    /*初始化*/
    ros::init(argc, argv, "yolo_detect");
    ros::NodeHandle nh;

    // /*加载yolo路径*/
    // std::string rospackPath = ros::package::getPath("yolov8_with_python");
    // yolo = new Yolov8(rospackPath, modelPath, classesPath);

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber imageSub = it.subscribe("test_camera", 100, imageCallback); // 订阅/cameraImage话题，并添加回调函数

    /*初始化Python*/
    Python_Initialize();

    ROS_INFO_GREEN("Waiting for image...");

    ros::spin(); // 循环等待回调函数触发

    // delete yolo;
}