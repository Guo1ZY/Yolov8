#include "Yolov8.hpp"

/*设置Numpy的版本*/
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#endif

#include <numpy/ndarrayobject.h>

#include <fstream>

/**
 * @brief 初始化Python环境，如果初始化失败，则会调用std::exit(1)退出程序
 *
 * @return 是否初始化成功
 */
int Python_Initialize()
{
    /*初始化Python*/
    Py_Initialize();

    /*检查Python是否初始化成功*/
    if (0 == Py_IsInitialized())
    {
        /*Python初始化失败*/
        COUT_RED_START;
        std::cerr << "Error!! Python Initialize Failed" << std::endl;
        COUT_COLOR_END;

        /*退出程序*/
        std::exit(1);
    }
    else
    {
        /*Python初始化成功*/
        COUT_GREEN_START;
        std::cout << "Python Initialize Success" << std::endl;
        COUT_COLOR_END;
    }

    /*初始化Numpy数组*/
    import_array1(-1);

    return 1;
}

/**
 * @brief 构造函数，同时初始化Yolov8模型和分类名
 *
 * @param rospackPath ROS包路径
 * @param modelPath Yolov8模型路径
 * @param classesPath 分类名字路径
 */
Yolov8::Yolov8(const std::string rospackPath, const std::string modelPath, const std::string classesPath)
{
    pySysPath = rospackPath + "/py/yolov8/py";

    /*加载Yolov8模型*/
    loadYolov8Model(modelPath);

    /*加载分类名*/
    loadClassesNames(classesPath);

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
    /*将Mat图片转为python array*/
    PyObject *pyArray = ConvertImageTopyArray(image);

    /*创建Python-tuple以便传参*/
    PyObject *pyArgs = Py_BuildValue("(O)", pyArray);

    /*调用Python Yolov8类的检测函数*/
    PyObject *pyResult = PyObject_Call(pyYolov8Detect, pyArgs, NULL);

    /*检查Python Yolov8类的检测函数是否调用成功，并转换为PyListObject*/
    if (pyResult == NULL || !PyList_Check(pyResult))
    {
        COUT_RED_START;
        std::cerr << "Error!! Python Yolov8 detect Failed" << std::endl;
        COUT_COLOR_END;

        std::vector<YoloDetect> zero;
        return zero;
    }

    /*转换数据格式*/
    std::vector<YoloDetect> results;

    Py_ssize_t numSublists = PyList_Size(pyResult);
    for (Py_ssize_t i = 0; i < numSublists; i++)
    {
        // 转换 list 形式的值
        PyObject *sublist = PyList_GetItem(pyResult, i);
        if (!PyList_Check(sublist) || PyList_Size(sublist) != 6)
        {
            std::cerr << "Error: Sublist has an invalid format." << std::endl;
            continue;
        }

        // 转换 xywh 形式的值
        float x = PyFloat_AsDouble(PyList_GetItem(sublist, 0));
        float y = PyFloat_AsDouble(PyList_GetItem(sublist, 1));
        float w = PyFloat_AsDouble(PyList_GetItem(sublist, 2));
        float h = PyFloat_AsDouble(PyList_GetItem(sublist, 3));

        // 转换 conf 形式的值
        float conf = PyFloat_AsDouble(PyList_GetItem(sublist, 4));

        // 转换 cls 形式的值
        float cls = PyFloat_AsDouble(PyList_GetItem(sublist, 5));

        // 转换到YoloDetect结构
        YoloDetect result;
        result.class_index = (int)cls;
        result.class_name = classes[(int)cls];
        result.confidence = conf;
        result.box = cv::Rect((int)(x - w / 2.0f), (int)(y - h / 2.0f), (int)w, (int)h);
        result.color = generateRandomColor();
        results.push_back(result);
    }

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
 * @brief 将Mat图片转为python array
 *
 * @param image Mat图片
 * @return PyObject* python array
 */
PyObject *Yolov8::ConvertImageTopyArray(const cv::Mat &image)
{
    /*将Mat图片转为python array*/
    npy_intp dims[3] = {image.rows, image.cols, image.channels()};
    PyObject *pyArray = PyArray_SimpleNewFromData(image.channels(), dims, NPY_UINT8, image.data);

    return pyArray;
}

/**
 * @brief 加载Yolov8模型
 *
 * @param modelPath Yolov8模型路径
 */
void Yolov8::loadYolov8Model(const std::string modelPath)
{
    /*为Python添加的import路径*/
    PyRun_SimpleString("import sys");
    std::string pyRun_Code = "sys.path.append('" + pySysPath + "')";
    PyRun_SimpleString(pyRun_Code.c_str());

    /*import导入的模块*/
    switch (yolomode)
    {
    case YoloMode::Pytorch:
        pyModule = PyImport_ImportModule("Yolov8");
        break;

    case YoloMode::Onnx:
        pyModule = PyImport_ImportModule("Yolov8_Onnx");
        break;

    default:
        break;
    }
    if (NULL == pyModule)
    {
        /*import导入失败*/
        COUT_RED_START;
        std::cerr << "Error!! Python Import Module Failed" << std::endl;
        COUT_COLOR_END;

        /*退出程序*/
        std::exit(1);
    }
    else
    {
        /*import导入成功*/
        COUT_GREEN_START;
        std::cout << "Python Import Module Success" << std::endl;
        COUT_COLOR_END;
    }

    /*提取Python类对象*/
    switch (yolomode)
    {
    case YoloMode::Pytorch:
        pyYolov8Class = PyObject_GetAttrString(pyModule, "Yolov8");
        break;

    case YoloMode::Onnx:
        pyYolov8Class = PyObject_GetAttrString(pyModule, "Yolov8_Onnx");
        break;

    default:
        break;
    }
    if (NULL == pyYolov8Class || 0 == PyCallable_Check(pyYolov8Class))
    {
        /*提取Python类对象失败*/
        COUT_RED_START;
        std::cerr << "Error!! Python Get Yolov8 Class Failed" << std::endl;
        COUT_COLOR_END;

        /*退出程序*/
        std::exit(1);
    }
    else
    {
        /*提取Python类对象成功*/
        COUT_GREEN_START;
        std::cout << "Python Get Yolov8 Class Success" << std::endl;
        COUT_COLOR_END;
    }

    /*调用类的构造函数的参数*/
    PyObject *pyArgs = PyTuple_New(3);                                   // 参数个数
    PyTuple_SetItem(pyArgs, 0, PyUnicode_FromString(modelPath.c_str())); // modelPath
    PyTuple_SetItem(pyArgs, 1, Py_BuildValue("f", confThreshold));       // confThreshold
    PyTuple_SetItem(pyArgs, 2, Py_BuildValue("i", cudaEnable));          // cudaEnable

    /*实例化Python类*/
    pyYolov8Instance = PyObject_CallObject(pyYolov8Class, pyArgs);

    /*提取Python Yolov8类的检测函数*/
    pyYolov8Detect = PyObject_GetAttrString(pyYolov8Instance, "detect");
    if (NULL == pyYolov8Detect || 0 == PyCallable_Check(pyYolov8Detect))
    {
        /*提取Python Yolov8类的检测函数失败*/
        COUT_RED_START;
        std::cerr << "Error!! Python Get Yolov8 detect Failed" << std::endl;
        COUT_COLOR_END;

        /*退出程序*/
        std::exit(1);
    }
    else
    {
        /*提取Python类对象成功*/
        COUT_GREEN_START;
        std::cout << "Python Get Yolov8 detect Success" << std::endl;
        COUT_COLOR_END;
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
 * @brief 生成随机颜色
 *
 * @return cv::Scalar 随机颜色
 */
cv::Scalar Yolov8::generateRandomColor()
{
    cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    return color;
}