
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

// 除了config.h外，你要修改的都在这了
//______________________________________________________________
std::string wts_name = "/home/zy/yolo11trt/best.wts";
std::string engine_name = "/home/zy/yolo11trt/best.wts.guo1zy"; // 模型后缀可以瞎写
std::string type = "n";
//--------------------------------------------------------------

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

int main()
{
    cudaSetDevice(kGpuId);
    int model_bboxes;
    float gd = 0, gw = 0;
    int max_channels = 0;
    if (type == "n")
    {
        gd = 0.50;
        gw = 0.25;
        max_channels = 1024;
    }
    else if (type == "s")
    {
        gd = 0.50;
        gw = 0.50;
        max_channels = 1024;
    }
    else if (type == "m")
    {
        gd = 0.50;
        gw = 1.00;
        max_channels = 512;
    }
    else if (type == "l")
    {
        gd = 1.0;
        gw = 1.0;
        max_channels = 512;
    }
    else if (type == "x")
    {
        gd = 1.0;
        gw = 1.50;
        max_channels = 512;
    }
    else
    {
        std::cout << "\033[31m" << "傻逼" << "\033[33m" << "你yolo种类选错了，type只有n,s,m,l,x" << "\033[0m" << std::endl;
        return -1;
    }
    // 生成trt推理引擎
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = nullptr;

    serialized_engine = buildEngineYolo11Det(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels, type);

    if (!serialized_engine)
    {
        std::cout << "\033[35m" << "模型生成失败！！！！" << std::endl;
        std::cout << "\033[31m" << "wts" << "\033[33m" << "选对了吗？" << std::endl;
        std::cout << "\033[31m" << "config.h" << "\033[33m" << "修改了吗？" << std::endl;
        std::cout << "\033[31m" << "MarkDomn" << "\033[33m" << "认真读了吗？" << std::endl;
        std::cout << "\033[31m" << "脑子" << "\033[1;31m" << "长了吗？" << "\033[0m" << std::endl;
        assert(false);
    }
    std::ofstream p(engine_name, std::ios::binary);
    if (!p)
    {
        std::cout << "\033[36m" << "?不是哥们，" << "\033[33m" << "选对路径" << "\033[31m" << "很难" << "\033[33m" << "吗?" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete serialized_engine;
    delete config;
    delete builder;
    std::cout << "\033[32m" << "模" << "\033[34m" << "型" << "\033[35m" << "构" << "\033[36m" << "建" << "\033[4;34m" << "成" << "\033[33m" << "功" << "\033[4;32m" << "喵" << "\033[4;35m" << "~\n请查阅:" << engine_name << "\033[0m" << std::endl;
    return 0;
}
