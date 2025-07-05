#include "main.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test");
    ros::NodeHandle ros_nh;

    image_transport::ImageTransport imagetransport(ros_nh);
    image_transport::Publisher color_image_pub;
    color_image_pub = imagetransport.advertise("test_camera", 100);

    cv::VideoCapture cap(0);
    while (ros::ok())
    {
        cv::Mat frame;
        cap >> frame;

        cv::resize(frame, frame, cv::Size(512, 512));

        ros::Time current_time = ros::Time::now();

        sensor_msgs::ImagePtr color_image_msg;
        color_image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        color_image_msg->header.frame_id = "camera";

        color_image_pub.publish(color_image_msg);

        uint32_t msec = current_time.sec * 1000 + current_time.nsec / 1000000;

        ROS_INFO("Current Time: %d milliseconds", msec);

        cv::imshow("src", frame);
        cv::waitKey(1);
    }
}