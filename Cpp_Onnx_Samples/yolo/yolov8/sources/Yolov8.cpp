#include "Yolov8.hpp"

/**
 * @brief 构造函数，同时初始化Yolov8模型和分类名
 *
 * @param modelPath Yolov8模型路径
 * @param classesPath 分类名字路径
 * @param runWithCuda 是否使用cuda
 * @param modelInputShape 模型输入尺寸
 * @param modelConfThreshold 置信度阈值
 * @param modelIouThreshold iou阈值
 */
Yolov8::Yolov8(const std::string modelPath, const std::string classesPath, const bool &runWithCuda, const cv::Size &modelInputShape, const float &modelConfThreshold, const float &modelIouThreshold)
{
    /*加载Yolov8模型*/
    loadYolov8Model(modelPath, runWithCuda);

    /*加载分类名*/
    loadClassesNames(classesPath);

    /*输入网络的尺寸*/
    inputShape = modelInputShape;

    /*置信度阈值*/
    confThreshold = modelConfThreshold;

    /*iou阈值*/
    iouThreshold = modelIouThreshold;

    /*创建随机数生成器*/
    rng = cv::RNG(cv::getTickCount());
}

/**
 * @brief Yolov8检测
 *
 * @param img 待检测图片
 * @return std::vector<YoloDetect> 检测结果
 */
std::vector<YoloDetect> Yolov8::detect(const cv::Mat &image)
{
    cv::Mat modelInput = image;

    /**将图片转换为正方形*/
    if (inputShape.width == inputShape.height)
        modelInput = formatToSquare(modelInput);

    /**将图片转换为blob*/
    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1 / 255.0f, inputShape, cv::Scalar(0, 0, 0), true, false);

    /**将blob传入网络*/
    net.setInput(blob);

    /**获取网络输出*/
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    /**处理网络输出*/
    std::vector<YoloDetect> results = postprocess(outputs, modelInput.cols, modelInput.rows);
    return results;
}

/**
 * @brief 绘制检测结果
 *
 * @param image 待绘制图片
 * @param result 检测结果
 */
void Yolov8::drawResult(cv::Mat &image, const std::vector<YoloDetect> &results)
{
    for (YoloDetect result : results)
    {
        // Draw a rectangle displaying the bounding box
        cv::rectangle(image, result.box, result.color, 2);

        // Get the label for the class name and its confidence
        std::string label = result.class_name + ":" + cv::format("%.2f", result.confidence);

        // Display the label at the top of the bounding box
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseLine);
        int top = std::max(result.box.y, labelSize.height);
        cv::putText(image, label, result.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1, result.color, 2);
    }
}

/**
 * @brief 加载Yolov8模型
 *
 * @param modelPath Yolov8模型路径
 * @param runWithCuda 是否使用cuda
 */
void Yolov8::loadYolov8Model(const std::string &modelPath, const bool &runWithCuda)
{
    net = cv::dnn::readNetFromONNX(modelPath);
    if (runWithCuda)
    {
        std::cout << "\nRunning on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\nRunning on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

/**
 * @brief 加载分类名字
 *
 * @param classesPath 分类名字路径
 */
void Yolov8::loadClassesNames(const std::string classesPath)
{
    /*读取分类名字*/
    std::ifstream classNamesFile(classesPath.c_str());
    if (classNamesFile.is_open())
    {
        /*读取分类名字成功*/
        COUT_GREEN_START;
        std::cout << "Open " << classesPath.c_str() << " File: ";
        std::string className = "";
        while (std::getline(classNamesFile, className))
        {
            classes.push_back(className);
            std::cout << className << " ";
        }
        std::cout << std::endl;
        COUT_COLOR_END;
    }
    else
    {
        /*读取分类名字失败*/
        COUT_RED_START;
        std::cerr << "Error!! Can not Open " << classesPath.c_str() << " File" << std::endl;
        COUT_COLOR_END;

        /*退出程序*/
        std::exit(1);
    }
}

/**
 * @brief 处理网络输出层数据
 *
 * @param outputs 网络输出层数据
 * @param image_cols 输入图片的宽
 * @param image_rows 输入图片的高
 * @return std::vector<YoloDetect> 处理后的检测结果
 */
std::vector<YoloDetect> Yolov8::postprocess(std::vector<cv::Mat> &outputs, const int image_cols, const int image_rows)
{
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    /*计算缩放比例*/
    float x_factor = (float)image_cols / (float)inputShape.width;
    float y_factor = (float)image_rows / (float)inputShape.height;

    /*获取输出层*/

    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];
    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    /*获取输出层的数据*/
    float *data = (float *)outputs[0].data;

    for (int i = 0; i < rows; i++, data += dimensions)
    {
        float *classes_scores = data + 4;

        /*获取最大类别的置信度*/
        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > confThreshold)
        {
            int centerX = (int)(data[0] * x_factor);
            int centerY = (int)(data[1] * y_factor);
            int width = (int)(data[2] * x_factor);
            int height = (int)(data[3] * y_factor);

            boxes.push_back(cv::Rect(centerX - 0.5 * width, centerY - 0.5 * height, width, height));
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
        }
    }

    /*非极大值抑制*/
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, nms_result);

    /*得到最后返回结果*/
    std::vector<YoloDetect> detections;
    for (size_t i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        YoloDetect result;
        result.class_index = class_ids[idx];
        result.confidence = confidences[idx];
        result.color = generateRandomColor();
        result.class_name = classes[result.class_index];
        result.box = boxes[idx];

        detections.push_back(result);
    }

    return detections;
}

/**
 * @brief 将图片转换为正方形
 *
 * @param source 待转换的图片
 * @return cv::Mat 正方形图片
 */
cv::Mat Yolov8::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

/**
 * @brief 生成随机颜色
 *
 * @return cv::Scalar 随机颜色
 */
cv::Scalar Yolov8::generateRandomColor()
{
    cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    return color;
}