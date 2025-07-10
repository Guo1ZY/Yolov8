#include <vector>
#include <random>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include "NvInferRuntimeCommon.h"
#include <string>

#ifndef __YOLOV8_HPP
#define __YOLOV8_HPP
#define COUT_RED_START std::cout << "\033[1;31m";
#define COUT_GREEN_START std::cout << "\033[1;32m";
#define COUT_YELLOW_START std::cout << "\033[1;33m";
#define COUT_BLUE_START std::cout << "\033[1;34m";
#define COUT_PURPLE_START std::cout << "\033[1;35m";
#define COUT_CYAN_START std::cout << "\033[1;36m";
#define COUT_WHITE_START std::cout << "\033[1;37m";
#define COUT_COLOR_END std::cout << "\033[0m";
#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess)                                                         \
        {                                                                                      \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#else

#if defined(_MSC_VER)
#define API __declspec(dllimport)
#else
#define API
#endif
#endif // API_EXPORTS

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

#endif // __MACROS_H

#ifndef TENSORRT_LOGGING_H
#define TENSORRT_LOGGING_H

using Severity = nvinfer1::ILogger::Severity;

class LogStreamConsumerBuffer : public std::stringbuf
{
public:
    LogStreamConsumerBuffer(std::ostream &stream, const std::string &prefix, bool shouldLog)
        : mOutput(stream), mPrefix(prefix), mShouldLog(shouldLog) {}

    LogStreamConsumerBuffer(LogStreamConsumerBuffer &&other) : mOutput(other.mOutput) {}

    ~LogStreamConsumerBuffer()
    {
        // std::streambuf::pbase() gives a pointer to the beginning of the buffered part of the output sequence
        // std::streambuf::pptr() gives a pointer to the current position of the output sequence
        // if the pointer to the beginning is not equal to the pointer to the current position,
        // call putOutput() to log the output to the stream
        if (pbase() != pptr())
        {
            putOutput();
        }
    }

    // synchronizes the stream buffer and returns 0 on success
    // synchronizing the stream buffer consists of inserting the buffer contents into the stream,
    // resetting the buffer and flushing the stream
    virtual int sync()
    {
        putOutput();
        return 0;
    }

    void putOutput()
    {
        if (mShouldLog)
        {
            // prepend timestamp
            std::time_t timestamp = std::time(nullptr);
            tm *tm_local = std::localtime(&timestamp);
            std::cout << "[";
            std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
            std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
            // std::stringbuf::str() gets the string contents of the buffer
            // insert the buffer contents pre-appended by the appropriate prefix into the stream
            mOutput << mPrefix << str();
            // set the buffer to empty
            str("");
            // flush the stream
            mOutput.flush();
        }
    }

    void setShouldLog(bool shouldLog) { mShouldLog = shouldLog; }

private:
    std::ostream &mOutput;
    std::string mPrefix;
    bool mShouldLog;
};

//!
//! \class LogStreamConsumerBase
//! \brief Convenience object used to initialize LogStreamConsumerBuffer before std::ostream in LogStreamConsumer
//!
class LogStreamConsumerBase
{
public:
    LogStreamConsumerBase(std::ostream &stream, const std::string &prefix, bool shouldLog)
        : mBuffer(stream, prefix, shouldLog) {}

protected:
    LogStreamConsumerBuffer mBuffer;
};

//!
//! \class LogStreamConsumer
//! \brief Convenience object used to facilitate use of C++ stream syntax when logging messages.
//!  Order of base classes is LogStreamConsumerBase and then std::ostream.
//!  This is because the LogStreamConsumerBase class is used to initialize the LogStreamConsumerBuffer member field
//!  in LogStreamConsumer and then the address of the buffer is passed to std::ostream.
//!  This is necessary to prevent the address of an uninitialized buffer from being passed to std::ostream.
//!  Please do not change the order of the parent classes.
//!
class LogStreamConsumer : protected LogStreamConsumerBase, public std::ostream
{
public:
    //! \brief Creates a LogStreamConsumer which logs messages with level severity.
    //!  Reportable severity determines if the messages are severe enough to be logged.
    LogStreamConsumer(Severity reportableSeverity, Severity severity)
        : LogStreamConsumerBase(severityOstream(severity), severityPrefix(severity), severity <= reportableSeverity),
          std::ostream(&mBuffer) // links the stream buffer with the stream
          ,
          mShouldLog(severity <= reportableSeverity),
          mSeverity(severity)
    {
    }

    LogStreamConsumer(LogStreamConsumer &&other)
        : LogStreamConsumerBase(severityOstream(other.mSeverity), severityPrefix(other.mSeverity), other.mShouldLog),
          std::ostream(&mBuffer) // links the stream buffer with the stream
          ,
          mShouldLog(other.mShouldLog),
          mSeverity(other.mSeverity)
    {
    }

    void setReportableSeverity(Severity reportableSeverity)
    {
        mShouldLog = mSeverity <= reportableSeverity;
        mBuffer.setShouldLog(mShouldLog);
    }

private:
    static std::ostream &severityOstream(Severity severity)
    {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    static std::string severityPrefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            return "[F] ";
        case Severity::kERROR:
            return "[E] ";
        case Severity::kWARNING:
            return "[W] ";
        case Severity::kINFO:
            return "[I] ";
        case Severity::kVERBOSE:
            return "[V] ";
        default:
            assert(0);
            return "";
        }
    }

    bool mShouldLog;
    Severity mSeverity;
};

//! \class Logger
//!
//! \brief Class which manages logging of TensorRT tools and samples
//!
//! \details This class provides a common interface for TensorRT tools and samples to log information to the console,
//! and supports logging two types of messages:
//!
//! - Debugging messages with an associated severity (info, warning, error, or internal error/fatal)
//! - Test pass/fail messages
//!
//! The advantage of having all samples use this class for logging as opposed to emitting directly to stdout/stderr is
//! that the logic for controlling the verbosity and formatting of sample output is centralized in one location.
//!
//! In the future, this class could be extended to support dumping test results to a file in some standard format
//! (for example, JUnit XML), and providing additional metadata (e.g. timing the duration of a test run).
//!
//! TODO: For backwards compatibility with existing samples, this class inherits directly from the nvinfer1::ILogger
//! interface, which is problematic since there isn't a clean separation between messages coming from the TensorRT
//! library and messages coming from the sample.
//!
//! In the future (once all samples are updated to use Logger::getTRTLogger() to access the ILogger) we can refactor the
//! class to eliminate the inheritance and instead make the nvinfer1::ILogger implementation a member of the Logger
//! object.

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING) : mReportableSeverity(severity) {}

    //!
    //! \enum TestResult
    //! \brief Represents the state of a given test
    //!
    enum class TestResult
    {
        kRUNNING, //!< The test is running
        kPASSED,  //!< The test passed
        kFAILED,  //!< The test failed
        kWAIVED   //!< The test was waived
    };

    //!
    //! \brief Forward-compatible method for retrieving the nvinfer::ILogger associated with this Logger
    //! \return The nvinfer1::ILogger associated with this Logger
    //!
    //! TODO Once all samples are updated to use this method to register the logger with TensorRT,
    //! we can eliminate the inheritance of Logger from ILogger
    //!
    nvinfer1::ILogger &getTRTLogger() { return *this; }

    //!
    //! \brief Implementation of the nvinfer1::ILogger::log() virtual method
    //!
    //! Note samples should not be calling this function directly; it will eventually go away once we eliminate the
    //! inheritance from nvinfer1::ILogger
    //!
    void log(Severity severity, const char *msg) TRT_NOEXCEPT override
    {
        LogStreamConsumer(mReportableSeverity, severity) << "[TRT] " << std::string(msg) << std::endl;
    }

    //!
    //! \brief Method for controlling the verbosity of logging output
    //!
    //! \param severity The logger will only emit messages that have severity of this level or higher.
    //!
    void setReportableSeverity(Severity severity) { mReportableSeverity = severity; }

    //!
    //! \brief Opaque handle that holds logging information for a particular test
    //!
    //! This object is an opaque handle to information used by the Logger to print test results.
    //! The sample must call Logger::defineTest() in order to obtain a TestAtom that can be used
    //! with Logger::reportTest{Start,End}().
    //!
    class TestAtom
    {
    public:
        TestAtom(TestAtom &&) = default;

    private:
        friend class Logger;

        TestAtom(bool started, const std::string &name, const std::string &cmdline)
            : mStarted(started), mName(name), mCmdline(cmdline) {}

        bool mStarted;
        std::string mName;
        std::string mCmdline;
    };

    //!
    //! \brief Define a test for logging
    //!
    //! \param[in] name The name of the test.  This should be a string starting with
    //!                  "TensorRT" and containing dot-separated strings containing
    //!                  the characters [A-Za-z0-9_].
    //!                  For example, "TensorRT.sample_googlenet"
    //! \param[in] cmdline The command line used to reproduce the test
    //
    //! \return a TestAtom that can be used in Logger::reportTest{Start,End}().
    //!
    static TestAtom defineTest(const std::string &name, const std::string &cmdline)
    {
        return TestAtom(false, name, cmdline);
    }

    //!
    //! \brief A convenience overloaded version of defineTest() that accepts an array of command-line arguments
    //!        as input
    //!
    //! \param[in] name The name of the test
    //! \param[in] argc The number of command-line arguments
    //! \param[in] argv The array of command-line arguments (given as C strings)
    //!
    //! \return a TestAtom that can be used in Logger::reportTest{Start,End}().
    static TestAtom defineTest(const std::string &name, int argc, char const *const *argv)
    {
        auto cmdline = genCmdlineString(argc, argv);
        return defineTest(name, cmdline);
    }

    //!
    //! \brief Report that a test has started.
    //!
    //! \pre reportTestStart() has not been called yet for the given testAtom
    //!
    //! \param[in] testAtom The handle to the test that has started
    //!
    static void reportTestStart(TestAtom &testAtom)
    {
        reportTestResult(testAtom, TestResult::kRUNNING);
        assert(!testAtom.mStarted);
        testAtom.mStarted = true;
    }

    //!
    //! \brief Report that a test has ended.
    //!
    //! \pre reportTestStart() has been called for the given testAtom
    //!
    //! \param[in] testAtom The handle to the test that has ended
    //! \param[in] result The result of the test. Should be one of TestResult::kPASSED,
    //!                   TestResult::kFAILED, TestResult::kWAIVED
    //!
    static void reportTestEnd(const TestAtom &testAtom, TestResult result)
    {
        assert(result != TestResult::kRUNNING);
        assert(testAtom.mStarted);
        reportTestResult(testAtom, result);
    }

    static int reportPass(const TestAtom &testAtom)
    {
        reportTestEnd(testAtom, TestResult::kPASSED);
        return EXIT_SUCCESS;
    }

    static int reportFail(const TestAtom &testAtom)
    {
        reportTestEnd(testAtom, TestResult::kFAILED);
        return EXIT_FAILURE;
    }

    static int reportWaive(const TestAtom &testAtom)
    {
        reportTestEnd(testAtom, TestResult::kWAIVED);
        return EXIT_SUCCESS;
    }

    static int reportTest(const TestAtom &testAtom, bool pass)
    {
        return pass ? reportPass(testAtom) : reportFail(testAtom);
    }

    Severity getReportableSeverity() const { return mReportableSeverity; }

private:
    //!
    //! \brief returns an appropriate string for prefixing a log message with the given severity
    //!
    static const char *severityPrefix(Severity severity)
    {
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            return "[F] ";
        case Severity::kERROR:
            return "[E] ";
        case Severity::kWARNING:
            return "[W] ";
        case Severity::kINFO:
            return "[I] ";
        case Severity::kVERBOSE:
            return "[V] ";
        default:
            assert(0);
            return "";
        }
    }

    //!
    //! \brief returns an appropriate string for prefixing a test result message with the given result
    //!
    static const char *testResultString(TestResult result)
    {
        switch (result)
        {
        case TestResult::kRUNNING:
            return "RUNNING";
        case TestResult::kPASSED:
            return "PASSED";
        case TestResult::kFAILED:
            return "FAILED";
        case TestResult::kWAIVED:
            return "WAIVED";
        default:
            assert(0);
            return "";
        }
    }

    //!
    //! \brief returns an appropriate output stream (cout or cerr) to use with the given severity
    //!
    static std::ostream &severityOstream(Severity severity)
    {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }

    //!
    //! \brief method that implements logging test results
    //!
    static void reportTestResult(const TestAtom &testAtom, TestResult result)
    {
        severityOstream(Severity::kINFO) << "&&&& " << testResultString(result) << " " << testAtom.mName << " # "
                                         << testAtom.mCmdline << std::endl;
    }

    //!
    //! \brief generate a command line string from the given (argc, argv) values
    //!
    static std::string genCmdlineString(int argc, char const *const *argv)
    {
        std::stringstream ss;
        for (int i = 0; i < argc; i++)
        {
            if (i > 0)
                ss << " ";
            ss << argv[i];
        }
        return ss.str();
    }

    Severity mReportableSeverity;
};

namespace
{

    //!
    //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kVERBOSE
    //!
    //! Example usage:
    //!
    //!     LOG_VERBOSE(logger) << "hello world" << std::endl;
    //!
    inline LogStreamConsumer LOG_VERBOSE(const Logger &logger)
    {
        return LogStreamConsumer(logger.getReportableSeverity(), Severity::kVERBOSE);
    }

    //!
    //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kINFO
    //!
    //! Example usage:
    //!
    //!     LOG_INFO(logger) << "hello world" << std::endl;
    //!
    inline LogStreamConsumer LOG_INFO(const Logger &logger)
    {
        return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINFO);
    }

    //!
    //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kWARNING
    //!
    //! Example usage:
    //!
    //!     LOG_WARN(logger) << "hello world" << std::endl;
    //!
    inline LogStreamConsumer LOG_WARN(const Logger &logger)
    {
        return LogStreamConsumer(logger.getReportableSeverity(), Severity::kWARNING);
    }

    //!
    //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kERROR
    //!
    //! Example usage:
    //!
    //!     LOG_ERROR(logger) << "hello world" << std::endl;
    //!
    inline LogStreamConsumer LOG_ERROR(const Logger &logger)
    {
        return LogStreamConsumer(logger.getReportableSeverity(), Severity::kERROR);
    }

    //!
    //! \brief produces a LogStreamConsumer object that can be used to log messages of severity kINTERNAL_ERROR
    //         ("fatal" severity)
    //!
    //! Example usage:
    //!
    //!     LOG_FATAL(logger) << "hello world" << std::endl;
    //!
    inline LogStreamConsumer LOG_FATAL(const Logger &logger)
    {
        return LogStreamConsumer(logger.getReportableSeverity(), Severity::kINTERNAL_ERROR);
    }

} // anonymous namespace
// Cpp native

using namespace nvinfer1;
/**
 * @brief Yolov8检测结果
 *
 */
struct YoloDetect
{
    int class_index = 0;                    // 类别索引
    std::string class_name = "";            // 类别名
    float confidence = 0.0;                 // 置信度
    cv::Rect box = cv::Rect(0, 0, 0, 0);    // 矩形
    cv::Scalar color = cv::Scalar(0, 0, 0); // 颜色
};
class Yolov8
{
public:
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
    Yolov8(const std::string modelPath, const std::string classesPath, const bool &runWithCuda = true, const cv::Size &modelInputShape = {256, 256}, const float &modelConfThreshold = 0.5, const float &modelIouThreshold = 0.5);
    /**
     * @brief Yolov8检测
     *
     * @param img 待检测图片
     * @return std::vector<YoloDetect> 检测结果
     */
    std::vector<YoloDetect> detect(cv::Mat &image);
    /**
     * @brief 绘制检测结果
     *
     * @param image 待绘制图片
     * @param result 检测结果
     */
    void drawResult(cv::Mat &image, const std::vector<YoloDetect> &results);

private:
    /*最大预测数，默认15，在Yolo.cpp中可以修改*/
    int kOutputSize;
    /*模型路径*/
    std::string engine_name;
    /*下面三个构建tensor引擎用*/
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;
    /*输出维度*/
    int model_bboxes;
    /*分配内存用的指针*/
    float *device_buffers[2];
    float *output_buffer_host;
    float *decode_ptr_host;
    float *decode_ptr_device;
    /*cuda流*/
    cudaStream_t stream;
    /*推理结果*/
    std::vector<YoloDetect> result;
    /*种类*/
    std::vector<std::string> classes;
    /*是否用gpu*/
    bool runWithCuda;
    /*模型推理尺寸，你训练时调的*/
    cv::Size img_size;
    /*置信度*/
    float modelConfThreshold;
    /*非极大值抑制阈值*/
    float modelIouThreshold;
    /*plugin插件注册*/
    Logger gLogger;
    /*标识名字，我也不知道有什么用*/
    char *kInputTensorName = "images";
    char *kOutputTensorName = "output";
    /*模型逆序列化（加载模型步骤）*/
    void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context);
    /*准备gpu内存*/
    void prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                        float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device, bool runWithCuda);
    /*推理*/
    void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, float *decode_ptr_host, float *decode_ptr_device, int model_bboxes, bool runWithCuda);
    /*加载模型名字*/
    void loadClassesNames(const std::string classesPath);
    /*把推理结果ip的尺寸转化为适应原图*/
    cv::Rect get_rect(const cv::Mat &img, const cv::Rect &bbox);
    /*iou检测，非极大值抑制用*/
    float iou(const cv::Rect &lbox, const cv::Rect &rbox);
    /*按照置信度排序*/
    static bool cmp(const YoloDetect &a, const YoloDetect &b);
    /*非极大值抑制*/
    void nms(std::vector<YoloDetect> &res, float *output, float conf_thresh, float nms_thresh, cv::Mat img);
    /*从gpu中导出推理结果*/
    void process_decode_ptr_host(std::vector<YoloDetect> &res, const float *decode_ptr_host, int bbox_element, cv::Mat &img, int count);
    void process_single_image(std::vector<YoloDetect> &res, const float *decode_ptr_host, int bbox_element, cv::Mat &img);
};

/*————————————————————————————————————————————————————————————————————
——————————————————————前面的区域以后再来探索吧————————————————————————————
——————————————————————————————————————————————————————————————————————*/
void cuda_preprocess_init(int max_image_size);

void cuda_preprocess_destroy();

void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height, cudaStream_t stream);

void cuda_batch_preprocess(cv::Mat &img_batch, float *dst, int dst_width, int dst_height, cudaStream_t stream);

void cuda_decode(float *predict, int num_bboxes, float confidence_threshold, float *parray, int max_objects,
                 cudaStream_t stream);

void cuda_nms(float *parray, float nms_threshold, int max_objects, cudaStream_t stream);

struct alignas(float) Detection
{
    // center_x center_y w h
    float bbox[4];
    float conf; // bbox_conf * cls_conf
    float class_id;
    float mask[32];
    float keypoints[51]; // 17*3 keypoints
};

struct AffineMatrix
{
    float value[6];
};

const int bbox_element =
    sizeof(AffineMatrix) / sizeof(float) + 1; // left, top, right, bottom, confidence, class, keepflag

namespace nvinfer1
{
    class API YoloLayerPlugin : public IPluginV2IOExt
    {
    public:
        YoloLayerPlugin(int classCount, int numberofpoints, float confthreshkeypoints, int netWidth, int netHeight,
                        int maxOut, bool is_segmentation, bool is_pose, const int *strides, int stridesLength);

        YoloLayerPlugin(const void *data, size_t length);

        ~YoloLayerPlugin();

        int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

        nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims) TRT_NOEXCEPT override;

        int initialize() TRT_NOEXCEPT override;

        virtual void terminate() TRT_NOEXCEPT override {}

        virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0; }

        virtual int enqueue(int batchSize, const void *const *inputs, void *TRT_CONST_ENQUEUE *outputs, void *workspace,
                            cudaStream_t stream) TRT_NOEXCEPT override;

        virtual size_t getSerializationSize() const TRT_NOEXCEPT override;

        virtual void serialize(void *buffer) const TRT_NOEXCEPT override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs,
                                       int nbOutputs) const TRT_NOEXCEPT override
        {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char *getPluginType() const TRT_NOEXCEPT override;

        const char *getPluginVersion() const TRT_NOEXCEPT override;

        void destroy() TRT_NOEXCEPT override;

        IPluginV2IOExt *clone() const TRT_NOEXCEPT override;

        void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override;

        const char *getPluginNamespace() const TRT_NOEXCEPT override;

        nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes,
                                             int32_t nbInputs) const TRT_NOEXCEPT;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                          int nbInputs) const TRT_NOEXCEPT override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

        void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                             IGpuAllocator *gpuAllocator) TRT_NOEXCEPT override;

        void configurePlugin(PluginTensorDesc const *in, int32_t nbInput, PluginTensorDesc const *out,
                             int32_t nbOutput) TRT_NOEXCEPT override;

        void detachFromContext() TRT_NOEXCEPT override;

    private:
        void forwardGpu(const float *const *inputs, float *output, cudaStream_t stream, int mYoloV8netHeight,
                        int mYoloV8NetWidth, int batchSize);

        int mThreadCount = 256;
        const char *mPluginNamespace;
        int mClassCount;
        int mNumberofpoints;
        float mConfthreshkeypoints;
        int mYoloV8NetWidth;
        int mYoloV8netHeight;
        int mMaxOutObject;
        bool is_segmentation_;
        bool is_pose_;
        int *mStrides;
        int mStridesLength;
    };

    class API YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char *getPluginName() const TRT_NOEXCEPT override;

        const char *getPluginVersion() const TRT_NOEXCEPT override;

        const nvinfer1::PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override;

        nvinfer1::IPluginV2IOExt *createPlugin(const char *name,
                                               const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT override;

        nvinfer1::IPluginV2IOExt *deserializePlugin(const char *name, const void *serialData,
                                                    size_t serialLength) TRT_NOEXCEPT override;

        void setPluginNamespace(const char *libNamespace) TRT_NOEXCEPT override { mNamespace = libNamespace; }

        const char *getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };

    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
} // namespace nvinfer1

#endif // YOLOV8_HPP