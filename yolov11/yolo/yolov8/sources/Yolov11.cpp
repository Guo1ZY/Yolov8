#include "Yolov11.hpp"

/**
 * @brief 构造函数，同时初始化Yolov11模型和分类名
 */
Yolov11::Yolov11(const std::string modelPath, const std::string classesPath, const bool &runWithCuda, const cv::Size &modelInputShape, const float &modelConfThreshold, const float &modelIouThreshold)
{
    /*加载Yolov11模型 - 放在最后，这样如果失败可以有更好的错误信息*/
    loadYolov11Model(modelPath, runWithCuda);

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

    // /*加载Yolov11模型 - 放在最后，这样如果失败可以有更好的错误信息*/
    // loadYolov11Model(modelPath, runWithCuda);
}

/**
 * @brief 加载Yolov11模型 - 增强兼容性版本
 */
void Yolov11::loadYolov11Model(const std::string &modelPath, const bool &runWithCuda)
{
    COUT_BLUE_START;
    std::cout << "Loading YOLOv11 model from: " << modelPath << std::endl;
    COUT_COLOR_END;

    try
    {
        // 方法1: 尝试标准加载方式
        net = cv::dnn::readNetFromONNX(modelPath);

        // 关键修复: 禁用所有可能导致问题的优化
        net.enableFusion(false);   // 禁用层融合
        net.enableWinograd(false); // 禁用Winograd优化

        COUT_GREEN_START;
        std::cout << "Model loaded successfully, configuring backend..." << std::endl;
        COUT_COLOR_END;

        // 配置后端 - 更保守的方式
        if (runWithCuda)
        {
            // COUT_BLUE_START;
            // std::cout << "Attempting CUDA backend..." << std::endl;
            // COUT_COLOR_END;

            try
            {
                // // 使用CUDA后端
                // net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                // net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

                // 测试推理以确保CUDA工作正常
                // testInference();

                // COUT_GREEN_START;
                // std::cout << "Running on cuda!" << std::endl;
                // COUT_COLOR_END;
                
            }
            catch (const std::exception &e)
            {
                COUT_RED_START;
                std::cout << "CUDA failed: " << e.what() << std::endl;
                std::cout << "Falling back to CPU..." << std::endl;
                COUT_COLOR_END;

                net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        }
        else
        {   
            std::cout << "Using CPU backend" << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }

        // 获取并验证输出层
        std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
        if (outNames.empty())
        {
            throw std::runtime_error("No output layers found in the model");
        }

        // std::cout << "Output layers (" << outNames.size() << "): ";
        // for (const auto &name : outNames)
        // {
        //     std::cout << name << " ";
        // }
        // std::cout << std::endl;

        // 最终测试
        // testInference();

        COUT_GREEN_START;
        std::cout << "-------------------" << std::endl;
        std::cout << "| Running on Cuda |" << std::endl;
        std::cout << "-------------------" << std::endl;
        COUT_COLOR_END;
        COUT_CYAN_START
        std::cout << "YOLOv11 model initialization completed successfully!" << std::endl;
        COUT_COLOR_END;
    }
    catch (const cv::Exception &e)
    {
        COUT_RED_START;
        std::cerr << "\n=== OpenCV DNN Error ===" << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "\n=== Possible Solutions ===" << std::endl;
        std::cerr << "1. Re-export your YOLOv11 model:" << std::endl;
        std::cerr << "   model.export(format='onnx', opset=11, simplify=True)" << std::endl;
        std::cerr << "2. Or use this Python script to fix the model:" << std::endl;
        std::cerr << "   python fix_yolov11_model.py --input best.onnx --output best_fixed.onnx" << std::endl;
        std::cerr << "3. Update OpenCV to version 4.7 for better YOLOv11 support" << std::endl;
        std::cerr << "4. Check if model file is corrupted" << std::endl;
        COUT_COLOR_END;
        throw;
    }
    catch (const std::exception &e)
    {
        COUT_RED_START;
        std::cerr << "General error: " << e.what() << std::endl;
        COUT_COLOR_END;
        throw;
    }
}

/**
 * @brief 测试推理以验证模型加载
 */
void Yolov11::testInference()
{
    try
    {
        // 创建测试输入 - 使用256x256
        cv::Mat testImg = cv::Mat::zeros(256, 256, CV_8UC3);
        cv::Mat blob;
        cv::dnn::blobFromImage(testImg, blob, 1.0 / 255.0, inputShape, cv::Scalar(0, 0, 0), true, false, CV_32F);

        // 进行测试推理
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        if (outputs.empty())
        {
            throw std::runtime_error("Model test inference failed - no outputs");
        }

        std::cout << "Test inference successful, output shape: ";
        for (int i = 0; i < outputs[0].dims; i++)
        {
            std::cout << outputs[0].size[i] << " ";
        }
        std::cout << std::endl;

        // 对于256x256输入，预期输出形状大约是 [1, 84, 2100]
        if (outputs[0].dims >= 3)
        {
            int expected_features = outputs[0].size[1];
            int num_detections = outputs[0].size[2];
            std::cout << "Features: " << expected_features << ", Detections: " << num_detections << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Test inference failed: " + std::string(e.what()));
    }
}

/**
 * @brief Yolov11检测 - 增强错误处理版本
 */
std::vector<YoloDetect> Yolov11::detect(const cv::Mat &image)
{
    if (image.empty())
    {
        std::cerr << "Warning: Empty image provided to detect()" << std::endl;
        return {};
    }

    try
    {
        cv::Mat modelInput = letterbox(image, inputShape);

        // 创建blob - 使用更安全的参数
        cv::Mat blob;
        cv::dnn::blobFromImage(modelInput, blob, 1.0 / 255.0, inputShape,
                               cv::Scalar(0, 0, 0), true, false, CV_32F);

        // 设置输入
        net.setInput(blob);

        // 前向推理
        std::vector<cv::Mat> outputs;
        try
        {
            net.forward(outputs, net.getUnconnectedOutLayersNames());
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "Inference error: " << e.what() << std::endl;
            return {};
        }

        // 后处理
        if (outputs.empty())
        {
            std::cerr << "Warning: No outputs from model inference" << std::endl;
            return {};
        }

        return postprocess(outputs, image.cols, image.rows);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Detection error: " << e.what() << std::endl;
        return {};
    }
}

/**
 * @brief 绘制检测结果
 */
void Yolov11::drawResult(cv::Mat &image, const std::vector<YoloDetect> &results)
{
    for (const YoloDetect &result : results)
    {
        // 确保边界框在图像范围内
        cv::Rect safeBox = result.box & cv::Rect(0, 0, image.cols, image.rows);
        if (safeBox.width <= 0 || safeBox.height <= 0)
            continue;

        // Draw bounding box
        cv::rectangle(image, safeBox, result.color, 2);

        // Prepare label
        std::string label = result.class_name + ":" + cv::format("%.2f", result.confidence);

        // Get text size
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // Ensure label position is within image
        int top = std::max(safeBox.y, labelSize.height);
        int left = safeBox.x;

        // Draw background rectangle for text
        cv::Rect textRect(left, top - labelSize.height, labelSize.width, labelSize.height + baseLine);
        textRect &= cv::Rect(0, 0, image.cols, image.rows);

        cv::rectangle(image, textRect, result.color, cv::FILLED);
        cv::putText(image, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

/**
 * @brief 加载分类名字
 */
void Yolov11::loadClassesNames(const std::string classesPath)
{
    std::ifstream classNamesFile(classesPath.c_str());
    if (classNamesFile.is_open())
    {
        COUT_GREEN_START;
        std::cout << "Loading classes from: " << classesPath << std::endl;
        COUT_COLOR_END

        COUT_CYAN_START
        std::cout << "Classes: ";
        COUT_COLOR_END

        COUT_PURPLE_START
        std::string className = "";
        while (std::getline(classNamesFile, className))
        {
            // 移除可能的回车符
            if (!className.empty() && className.back() == '\r')
            {
                className.pop_back();
            }
            if (!className.empty())
            {
                classes.push_back(className);
                std::cout << className << " ";
            }
        }
        std::cout << std::endl;
        COUT_COLOR_END
        std::cout << "Total classes loaded: " << classes.size() << std::endl;
    }
    else
    {
        COUT_RED_START;
        std::cerr << "Error!! Cannot open classes file: " << classesPath << std::endl;
        COUT_COLOR_END;
        std::exit(1);
    }
}

/**
 * @brief 处理网络输出层数据
 */
std::vector<YoloDetect> Yolov11::postprocess(std::vector<cv::Mat> &outputs, const int image_cols, const int image_rows)
{
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    if (outputs.empty())
    {
        return {};
    }

    cv::Mat output = outputs[0];

    // Debug output shape
    // std::cout << "Raw output shape: ";
    // for (int i = 0; i < output.dims; i++)
    // {
    //     std::cout << output.size[i] << " ";
    // }
    // std::cout << std::endl;

    // 计算缩放比例（letterbox）
    float gain = std::min((float)inputShape.width / image_cols, (float)inputShape.height / image_rows);
    float pad_x = (inputShape.width - image_cols * gain) / 2;
    float pad_y = (inputShape.height - image_rows * gain) / 2;

    // 处理不同的输出格式
    int num_classes = classes.size();
    int num_boxes = 0;
    int output_dims = 0;

    if (output.dims == 3)
    {
        // 格式: [1, 84, num_boxes]
        // 对于256x256输入，num_boxes通常是2100左右
        // 对于640x640输入，num_boxes通常是8400
        num_boxes = output.size[2];
        output_dims = output.size[1];

        // Transpose: [1, 84, num_boxes] -> [num_boxes, 84]
        output = output.reshape(1, output_dims);
        cv::transpose(output, output);
    }
    else if (output.dims == 2)
    {
        // 格式: [num_boxes, 84]
        num_boxes = output.size[0];
        output_dims = output.size[1];
    }
    else
    {
        std::cerr << "Unsupported output format, dims: " << output.dims << std::endl;
        return {};
    }

    // std::cout << "Processed: " << num_boxes << " boxes, " << output_dims << " dimensions" << std::endl;
    // std::cout << "Expected classes: " << num_classes << std::endl;

    // 验证输出维度
    if (output_dims < 4 + num_classes)
    {
        std::cerr << "Error: Output dimensions (" << output_dims << ") insufficient for "
                  << num_classes << " classes" << std::endl;
        return {};
    }

    float *data = (float *)output.data;

    for (int i = 0; i < num_boxes; i++, data += output_dims)
    {
        // 提取边界框坐标 (center_x, center_y, width, height)
        float center_x = data[0];
        float center_y = data[1];
        float width = data[2];
        float height = data[3];

        // 提取类别分数
        float *class_scores = data + 4;

        // 找到最高分数的类别
        cv::Mat scores(1, num_classes, CV_32FC1, class_scores);
        cv::Point class_id;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        if (max_class_score > confThreshold)
        {
            // 将坐标转换回原图空间
            center_x = (center_x - pad_x) / gain;
            center_y = (center_y - pad_y) / gain;
            width = width / gain;
            height = height / gain;

            // 计算左上角坐标
            int left = (int)(center_x - width / 2);
            int top = (int)(center_y - height / 2);

            // 确保坐标在图像范围内
            left = std::max(0, std::min(left, image_cols - 1));
            top = std::max(0, std::min(top, image_rows - 1));
            width = std::min(width, (float)(image_cols - left));
            height = std::min(height, (float)(image_rows - top));

            if (width > 0 && height > 0)
            {
                boxes.push_back(cv::Rect(left, top, (int)width, (int)height));
                confidences.push_back(max_class_score);
                class_ids.push_back(class_id.x);
            }
        }
    }

    // std::cout << "Found " << boxes.size() << " detections before NMS" << std::endl;

    // 非极大值抑制
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, nms_result);

    // std::cout << "After NMS: " << nms_result.size() << " detections" << std::endl;

    // 构建最终结果
    std::vector<YoloDetect> detections;
    for (size_t i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        if (class_ids[idx] >= 0 && class_ids[idx] < classes.size())
        {
            YoloDetect result;
            result.class_index = class_ids[idx];
            result.confidence = confidences[idx];
            result.color = generateRandomColor();
            result.class_name = classes[result.class_index];
            result.box = boxes[idx];
            detections.push_back(result);
        }
    }

    return detections;
}

/**
 * @brief letterbox预处理
 */
cv::Mat Yolov11::letterbox(const cv::Mat &image, const cv::Size &newShape)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);

    cv::Size newUnpad = cv::Size((int)std::round((float)shape.width * r),
                                 (int)std::round((float)shape.height * r));

    int dw = newShape.width - newUnpad.width;
    int dh = newShape.height - newUnpad.height;

    dw /= 2;
    dh /= 2;

    cv::Mat resized;
    if (shape.width != newUnpad.width || shape.height != newUnpad.height)
    {
        cv::resize(image, resized, newUnpad, 0, 0, cv::INTER_LINEAR);
    }
    else
    {
        resized = image.clone();
    }

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, dh, dh, dw, dw,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return padded;
}

/**
 * @brief 生成随机颜色
 */
cv::Scalar Yolov11::generateRandomColor()
{
    // return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    return cv::Scalar(0,255,0);
}