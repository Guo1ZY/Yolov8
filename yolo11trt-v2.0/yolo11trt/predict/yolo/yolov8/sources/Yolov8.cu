#include <assert.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "Yolov8.hpp"

namespace Tn
{
    template <typename T>
    void write(char *&buffer, const T &val)
    {
        *reinterpret_cast<T *>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    void read(const char *&buffer, T &val)
    {
        val = *reinterpret_cast<const T *>(buffer);
        buffer += sizeof(T);
    }
} // namespace Tn

__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin(int classCount, int numberofpoints, float confthreshkeypoints, int netWidth,
                                     int netHeight, int maxOut, bool is_segmentation, bool is_pose, const int *strides,
                                     int stridesLength)
    {

        mClassCount = classCount;
        mNumberofpoints = numberofpoints;
        mConfthreshkeypoints = confthreshkeypoints;
        mYoloV8NetWidth = netWidth;
        mYoloV8netHeight = netHeight;
        mMaxOutObject = maxOut;
        mStridesLength = stridesLength;
        mStrides = new int[stridesLength];
        memcpy(mStrides, strides, stridesLength * sizeof(int));
        is_segmentation_ = is_segmentation;
        is_pose_ = is_pose;
    }

    YoloLayerPlugin::~YoloLayerPlugin()
    {
        if (mStrides != nullptr)
        {
            delete[] mStrides;
            mStrides = nullptr;
        }
    }

    YoloLayerPlugin::YoloLayerPlugin(const void *data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mNumberofpoints);
        read(d, mConfthreshkeypoints);
        read(d, mThreadCount);
        read(d, mYoloV8NetWidth);
        read(d, mYoloV8netHeight);
        read(d, mMaxOutObject);
        read(d, mStridesLength);
        mStrides = new int[mStridesLength];
        for (int i = 0; i < mStridesLength; ++i)
        {
            read(d, mStrides[i]);
        }
        read(d, is_segmentation_);
        read(d, is_pose_);

        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void *buffer) const TRT_NOEXCEPT
    {

        using namespace Tn;
        char *d = static_cast<char *>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mNumberofpoints);
        write(d, mConfthreshkeypoints);
        write(d, mThreadCount);
        write(d, mYoloV8NetWidth);
        write(d, mYoloV8netHeight);
        write(d, mMaxOutObject);
        write(d, mStridesLength);
        for (int i = 0; i < mStridesLength; ++i)
        {
            write(d, mStrides[i]);
        }
        write(d, is_segmentation_);
        write(d, is_pose_);

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const TRT_NOEXCEPT
    {
        return sizeof(mClassCount) + sizeof(mNumberofpoints) + sizeof(mConfthreshkeypoints) + sizeof(mThreadCount) +
               sizeof(mYoloV8netHeight) + sizeof(mYoloV8NetWidth) + sizeof(mMaxOutObject) + sizeof(mStridesLength) +
               sizeof(int) * mStridesLength + sizeof(is_segmentation_) + sizeof(is_pose_);
    }

    int YoloLayerPlugin::initialize() TRT_NOEXCEPT
    {
        return 0;
    }

    nvinfer1::Dims YoloLayerPlugin::getOutputDimensions(int index, const nvinfer1::Dims *inputs,
                                                        int nbInputDims) TRT_NOEXCEPT
    {
        int total_size = mMaxOutObject * sizeof(Detection) / sizeof(float);
        return nvinfer1::Dims3(total_size + 1, 1, 1);
    }

    void YoloLayerPlugin::setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT
    {
        mPluginNamespace = pluginNamespace;
    }

    const char *YoloLayerPlugin::getPluginNamespace() const TRT_NOEXCEPT
    {
        return mPluginNamespace;
    }

    nvinfer1::DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                          int nbInputs) const TRT_NOEXCEPT
    {
        return nvinfer1::DataType::kFLOAT;
    }

    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                                       int nbInputs) const TRT_NOEXCEPT
    {

        return false;
    }

    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
    {

        return false;
    }

    void YoloLayerPlugin::configurePlugin(nvinfer1::PluginTensorDesc const *in, int nbInput,
                                          nvinfer1::PluginTensorDesc const *out, int nbOutput) TRT_NOEXCEPT {};

    void YoloLayerPlugin::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext,
                                          IGpuAllocator *gpuAllocator) TRT_NOEXCEPT {};

    void YoloLayerPlugin::detachFromContext() TRT_NOEXCEPT {}

    const char *YoloLayerPlugin::getPluginType() const TRT_NOEXCEPT
    {

        return "YoloLayer_TRT";
    }

    const char *YoloLayerPlugin::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    void YoloLayerPlugin::destroy() TRT_NOEXCEPT
    {
        delete this;
    }

    nvinfer1::IPluginV2IOExt *YoloLayerPlugin::clone() const TRT_NOEXCEPT
    {

        YoloLayerPlugin *p =
            new YoloLayerPlugin(mClassCount, mNumberofpoints, mConfthreshkeypoints, mYoloV8NetWidth, mYoloV8netHeight,
                                mMaxOutObject, is_segmentation_, is_pose_, mStrides, mStridesLength);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    int YoloLayerPlugin::enqueue(int batchSize, const void *TRT_CONST_ENQUEUE *inputs, void *const *outputs,
                                 void *workspace, cudaStream_t stream) TRT_NOEXCEPT
    {
        forwardGpu((const float *const *)inputs, (float *)outputs[0], stream, mYoloV8netHeight, mYoloV8NetWidth, batchSize);
        return 0;
    }

    __device__ float Logist(float data)
    {
        return 1.0f / (1.0f + expf(-data));
    };

    __global__ void CalDetection(const float *input, float *output, int numElements, int maxoutobject, const int grid_h,
                                 int grid_w, const int stride, int classes, int nk, float confkeypoints, int outputElem,
                                 bool is_segmentation, bool is_pose)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= numElements)
            return;

        const int N_kpts = nk;
        int total_grid = grid_h * grid_w;
        int info_len = 4 + classes + (is_segmentation ? 32 : 0) + (is_pose ? N_kpts * 3 : 0);
        int batchIdx = idx / total_grid;
        int elemIdx = idx % total_grid;
        const float *curInput = input + batchIdx * total_grid * info_len;
        int outputIdx = batchIdx * outputElem;

        int class_id = 0;
        float max_cls_prob = 0.0;
        for (int i = 4; i < 4 + classes; i++)
        {
            float p = Logist(curInput[elemIdx + i * total_grid]);
            if (p > max_cls_prob)
            {
                max_cls_prob = p;
                class_id = i - 4;
            }
        }

        if (max_cls_prob < 0.1)
            return;

        int count = (int)atomicAdd(output + outputIdx, 1);
        if (count >= maxoutobject)
            return;
        char *data = (char *)(output + outputIdx) + sizeof(float) + count * sizeof(Detection);
        Detection *det = (Detection *)(data);

        int row = elemIdx / grid_w;
        int col = elemIdx % grid_w;

        det->conf = max_cls_prob;
        det->class_id = class_id;
        det->bbox[0] = (col + 0.5f - curInput[elemIdx + 0 * total_grid]) * stride;
        det->bbox[1] = (row + 0.5f - curInput[elemIdx + 1 * total_grid]) * stride;
        det->bbox[2] = (col + 0.5f + curInput[elemIdx + 2 * total_grid]) * stride;
        det->bbox[3] = (row + 0.5f + curInput[elemIdx + 3 * total_grid]) * stride;

        if (is_segmentation)
        {
            for (int k = 0; k < 32; ++k)
            {
                det->mask[k] = curInput[elemIdx + (4 + classes + k) * total_grid];
            }
        }

        if (is_pose)
        {
            for (int kpt = 0; kpt < N_kpts; kpt++)
            {
                int kpt_x_idx = (4 + classes + (is_segmentation ? 32 : 0) + kpt * 3) * total_grid;
                int kpt_y_idx = (4 + classes + (is_segmentation ? 32 : 0) + kpt * 3 + 1) * total_grid;
                int kpt_conf_idx = (4 + classes + (is_segmentation ? 32 : 0) + kpt * 3 + 2) * total_grid;

                float kpt_confidence = sigmoid(curInput[elemIdx + kpt_conf_idx]);

                float kpt_x = (curInput[elemIdx + kpt_x_idx] * 2.0 + col) * stride;
                float kpt_y = (curInput[elemIdx + kpt_y_idx] * 2.0 + row) * stride;

                bool is_within_bbox =
                    kpt_x >= det->bbox[0] && kpt_x <= det->bbox[2] && kpt_y >= det->bbox[1] && kpt_y <= det->bbox[3];

                if (kpt_confidence < confkeypoints || !is_within_bbox)
                {
                    det->keypoints[kpt * 3] = -1;
                    det->keypoints[kpt * 3 + 1] = -1;
                    det->keypoints[kpt * 3 + 2] = -1;
                }
                else
                {
                    det->keypoints[kpt * 3] = kpt_x;
                    det->keypoints[kpt * 3 + 1] = kpt_y;
                    det->keypoints[kpt * 3 + 2] = kpt_confidence;
                }
            }
        }
    }

    void YoloLayerPlugin::forwardGpu(const float *const *inputs, float *output, cudaStream_t stream, int mYoloV8netHeight,
                                     int mYoloV8NetWidth, int batchSize)
    {
        int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
        cudaMemsetAsync(output, 0, sizeof(float), stream);
        for (int idx = 0; idx < batchSize; ++idx)
        {
            CUDA_CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
        }
        int numElem = 0;

        //    const int maxGrids = mStridesLength;
        //    int grids[maxGrids][2];
        //    for (int i = 0; i < maxGrids; ++i) {
        //        grids[i][0] = mYoloV8netHeight / mStrides[i];
        //        grids[i][1] = mYoloV8NetWidth / mStrides[i];
        //    }

        int maxGrids = mStridesLength;
        int flatGridsLen = 2 * maxGrids;
        int *flatGrids = new int[flatGridsLen];

        for (int i = 0; i < maxGrids; ++i)
        {
            flatGrids[2 * i] = mYoloV8netHeight / mStrides[i];
            flatGrids[2 * i + 1] = mYoloV8NetWidth / mStrides[i];
        }

        for (unsigned int i = 0; i < maxGrids; i++)
        {
            // Access the elements of the original 2D array from the flattened 1D array
            int grid_h = flatGrids[2 * i];     // Corresponds to the access of grids[i][0]
            int grid_w = flatGrids[2 * i + 1]; // Corresponds to the access of grids[i][1]
            int stride = mStrides[i];
            numElem = grid_h * grid_w * batchSize; // Calculate the total number of elements
            if (numElem < mThreadCount)            // Adjust the thread count if needed
                mThreadCount = numElem;

            // The CUDA kernel call remains unchanged
            CalDetection<<<(numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream>>>(
                inputs[i], output, numElem, mMaxOutObject, grid_h, grid_w, stride, mClassCount, mNumberofpoints,
                mConfthreshkeypoints, outputElem, is_segmentation_, is_pose_);
        }

        delete[] flatGrids;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char *YoloPluginCreator::getPluginName() const TRT_NOEXCEPT
    {
        return "YoloLayer_TRT";
    }

    const char *YoloPluginCreator::getPluginVersion() const TRT_NOEXCEPT
    {
        return "1";
    }

    const PluginFieldCollection *YoloPluginCreator::getFieldNames() TRT_NOEXCEPT
    {
        return &mFC;
    }

    IPluginV2IOExt *YoloPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) TRT_NOEXCEPT
    {
        assert(fc->nbFields == 1);
        assert(strcmp(fc->fields[0].name, "combinedInfo") == 0);
        const int *combinedInfo = static_cast<const int *>(fc->fields[0].data);
        int netinfo_count = 8;
        int class_count = combinedInfo[0];
        int numberofpoints = combinedInfo[1];
        float confthreshkeypoints = combinedInfo[2];
        int input_w = combinedInfo[3];
        int input_h = combinedInfo[4];
        int max_output_object_count = combinedInfo[5];
        bool is_segmentation = combinedInfo[6];
        bool is_pose = combinedInfo[7];
        const int *px_arry = combinedInfo + netinfo_count;
        int px_arry_length = fc->fields[0].length - netinfo_count;
        YoloLayerPlugin *obj =
            new YoloLayerPlugin(class_count, numberofpoints, confthreshkeypoints, input_w, input_h,
                                max_output_object_count, is_segmentation, is_pose, px_arry, px_arry_length);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt *YoloPluginCreator::deserializePlugin(const char *name, const void *serialData,
                                                         size_t serialLength) TRT_NOEXCEPT
    {
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        YoloLayerPlugin *obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

} // namespace nvinfer1

static uint8_t *img_buffer_host = nullptr;
static uint8_t *img_buffer_device = nullptr;

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
    : engine_name(modelPath), runtime(nullptr),
      engine(nullptr), context(nullptr), output_buffer_host(nullptr), decode_ptr_host(nullptr), decode_ptr_device(nullptr), runWithCuda(runWithCuda), img_size(modelInputShape), modelConfThreshold(modelConfThreshold), modelIouThreshold(modelIouThreshold)
{
    loadClassesNames(classesPath);
    kOutputSize = 15 * sizeof(YoloDetect) / sizeof(float) + 1;
    cudaSetDevice(0);
    deserialize_engine(engine_name, &runtime, &engine, &context);
    CUDA_CHECK(cudaStreamCreate(&stream));
    // 3000*3000 is the max size of the image you input
    cuda_preprocess_init(3000 * 3000);
    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];
    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host, &decode_ptr_device, runWithCuda);

    // 玄学，大幅降低后面推理的延迟，不信你把我删掉试试
    cv::Mat blank_image = cv::Mat::zeros(1, 1, CV_8UC3);
    detect(blank_image);
}

/**
 * @brief Yolov8检测
 *
 * @param img 待检测图片
 * @return std::vector<YoloDetect> 检测结果
 */
std::vector<YoloDetect> Yolov8::detect(cv::Mat &image)
{
    // Preprocess
    cuda_batch_preprocess(image, device_buffers[0], img_size.width, img_size.height, stream);
    // Run inference
    infer(*context, stream, (void **)device_buffers, output_buffer_host, decode_ptr_host, decode_ptr_device, model_bboxes, runWithCuda);
    std::vector<YoloDetect> res_batch;
    if (runWithCuda == 0)
    {
        // NMS
        nms(res_batch, output_buffer_host, modelIouThreshold, modelIouThreshold, image);
    }
    else
    {
        // Process gpu decode and nms results
        process_single_image(res_batch, decode_ptr_host, bbox_element, image);
    }
    // Draw bounding boxes
    return res_batch;
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

void Yolov8::deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context)
{
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}
void Yolov8::prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                            float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device, bool runWithCuda)
{
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void **)input_buffer_device, 3 * img_size.height * img_size.width * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)output_buffer_device, kOutputSize * sizeof(float)));
    if (!runWithCuda)
    {
        *output_buffer_host = new float[kOutputSize];
    }
    else
    {
        // Allocate memory for decode_ptr_host and copy to device
        *decode_ptr_host = new float[1 + kOutputSize * bbox_element];
        CUDA_CHECK(cudaMalloc((void **)decode_ptr_device, sizeof(float) * (1 + kOutputSize * bbox_element)));
    }
}

void Yolov8::infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, float *decode_ptr_host, float *decode_ptr_device, int model_bboxes, bool runWithCuda)
{
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueueV2(buffers, stream, nullptr);
    if (runWithCuda == 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream));
    }
    else
    {
        CUDA_CHECK(
            cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kOutputSize * bbox_element), stream));
        cuda_decode((float *)buffers[1], model_bboxes, modelConfThreshold, decode_ptr_device, kOutputSize, stream);
        cuda_nms(decode_ptr_device, modelIouThreshold, kOutputSize, stream); // cuda nms
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                   sizeof(float) * (1 + kOutputSize * bbox_element), cudaMemcpyDeviceToHost,
                                   stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

cv::Rect Yolov8::get_rect(const cv::Mat &img, const cv::Rect &bbox)
{
    float l, r, t, b;
    float r_w = img_size.width / (img.cols * 1.0);
    float r_h = img_size.height / (img.rows * 1.0);

    if (r_h > r_w)
    {
        l = bbox.x;
        r = bbox.x + bbox.width;
        t = bbox.y - (img_size.height - r_w * img.rows) / 2;
        b = bbox.y + bbox.height - (img_size.height - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else
    {
        l = bbox.x - (img_size.width - r_h * img.cols) / 2;
        r = bbox.x + bbox.width - (img_size.width - r_h * img.cols) / 2;
        t = bbox.y;
        b = bbox.y + bbox.height;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

float Yolov8::iou(const cv::Rect &lbox, const cv::Rect &rbox)
{
    float interBox[] = {
        static_cast<float>((std::max)(lbox.x, rbox.x)),                            // left
        static_cast<float>((std::min)(lbox.x + lbox.width, rbox.x + rbox.width)),  // right
        static_cast<float>((std::max)(lbox.y, rbox.y)),                            // top
        static_cast<float>((std::min)(lbox.y + lbox.height, rbox.y + rbox.height)) // bottom
    };
    if (interBox[2] >= interBox[3] || interBox[0] >= interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    float unionBoxS = static_cast<float>(lbox.width * lbox.height + rbox.width * rbox.height - interBoxS);

    return interBoxS / unionBoxS;
}

bool Yolov8::cmp(const YoloDetect &a, const YoloDetect &b)
{
    return a.confidence > b.confidence;
}

void Yolov8::nms(std::vector<YoloDetect> &res, float *output, float conf_thresh, float nms_thresh, cv::Mat img)
{
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<YoloDetect>> m;
    for (int i = 0; i < output[0]; i++)
    {
        if (output[1 + det_size * i + 4] <= conf_thresh)
            continue;
        YoloDetect det;
        // 解析并赋值 box (x1, y1, x2, y2 -> cv::Rect)
        det.box.x = static_cast<int>(output[1 + det_size * i]);
        det.box.y = static_cast<int>(output[1 + det_size * i + 1]);
        det.box.width = static_cast<int>(output[1 + det_size * i + 2] - det.box.x);
        det.box.height = static_cast<int>(output[1 + det_size * i + 3] - det.box.y);
        det.box = get_rect(img, det.box);
        // 赋值 confidence 和 class_index
        det.confidence = output[1 + det_size * i + 4];
        det.class_index = static_cast<int>(output[1 + det_size * i + 5]);
        det.class_name = classes[det.class_index];
        // 优化了光敏性癫痫患者的使用体验
        det.color = cv::Scalar(255 * sin(180 * float(det.class_index) / classes.size()), 255 * cos(180 * float(det.class_index) / classes.size()), 255 * (float(det.class_index) / classes.size()));
        if (m.count(det.class_index) == 0)
            m.emplace(det.class_index, std::vector<YoloDetect>());
        m[det.class_index].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++)
    {
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m)
        {
            auto &item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n)
            {
                if (iou(item.box, dets[n].box) > nms_thresh)
                {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void Yolov8::process_decode_ptr_host(std::vector<YoloDetect> &res, const float *decode_ptr_host, int bbox_element, cv::Mat &img, int count)
{
    YoloDetect det;
    for (int i = 0; i < count; i++)
    {
        int basic_pos = 1 + i * bbox_element;
        int keep_flag = decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1)
        {
            det.box.x = decode_ptr_host[basic_pos + 0];
            det.box.y = decode_ptr_host[basic_pos + 1];
            det.box.width = decode_ptr_host[basic_pos + 2] - det.box.x;
            det.box.height = decode_ptr_host[basic_pos + 3] - det.box.y;
            det.confidence = decode_ptr_host[basic_pos + 4];
            det.class_index = decode_ptr_host[basic_pos + 5];
            det.box = get_rect(img, det.box);
            det.class_name = classes[det.class_index];
            // 优化了光敏性癫痫患者的使用体验
            det.color = cv::Scalar(255 * sin(180 * float(det.class_index) / classes.size()), 255 * cos(180 * float(det.class_index) / classes.size()), 255 * (float(det.class_index) / classes.size()));

            // 检查是否已存在相同的 box 值，这是nms的bug，在cuda编程部分里，我以后再去优化吧，少不了几微秒
            bool duplicate = false;
            for (const auto &existing_det : res)
            {
                if (existing_det.box == det.box)
                {
                    duplicate = true;
                    break;
                }
            }

            // 如果没有重复，将该检测框加入结果集
            if (!duplicate)
            {
                res.push_back(det);
            }
        }
    }
}

void Yolov8::process_single_image(std::vector<YoloDetect> &res, const float *decode_ptr_host, int bbox_element, cv::Mat &img)
{
    int count = static_cast<int>(*decode_ptr_host);
    count = std::min(count, 15);

    process_decode_ptr_host(res, decode_ptr_host, bbox_element, img, count);
}

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> WeightMap;

    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; x++)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        wt.count = size;
        WeightMap[name] = wt;
    }
    return WeightMap;
}

__global__ void warpaffine_kernel(uint8_t *src, int src_line_size, int src_width, int src_height, float *dst,
                                  int dst_width, int dst_height, uint8_t const_value_st, AffineMatrix d2s, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge)
        return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height)
    {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else
    {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t *v1 = const_value;
        uint8_t *v2 = const_value;
        uint8_t *v3 = const_value;
        uint8_t *v4 = const_value;

        if (y_low >= 0)
        {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height)
        {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // bgr to rgb
    float t = c2;
    c2 = c0;
    c0 = t;

    // normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float *pdst_c0 = dst + dy * dst_width + dx;
    float *pdst_c1 = pdst_c0 + area;
    float *pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void cuda_preprocess(uint8_t *src, int src_width, int src_height, float *dst, int dst_width, int dst_height,
                     cudaStream_t stream)
{
    int img_size = src_width * src_height * 3;
    // copy data to pinned memory
    memcpy(img_buffer_host, src, img_size);
    // copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream));

    AffineMatrix s2d, d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;
    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);
    warpaffine_kernel<<<blocks, threads, 0, stream>>>(img_buffer_device, src_width * 3, src_width, src_height, dst,
                                                      dst_width, dst_height, 128, d2s, jobs);
}

void cuda_batch_preprocess(cv::Mat &img_batch,
                           float *dst, int dst_width, int dst_height,
                           cudaStream_t stream)
{
    int dst_size = dst_width * dst_height * 3;

    cuda_preprocess(img_batch.ptr(), img_batch.cols, img_batch.rows, &dst[0], dst_width,
                    dst_height, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void cuda_preprocess_init(int max_image_size)
{
    // prepare input data in pinned memory
    CUDA_CHECK(cudaMallocHost((void **)&img_buffer_host, max_image_size * 3));
    // prepare input data in device memory
    CUDA_CHECK(cudaMalloc((void **)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy()
{
    CUDA_CHECK(cudaFree(img_buffer_device));
    CUDA_CHECK(cudaFreeHost(img_buffer_host));
}

static __global__ void decode_kernel(float *predict, int num_bboxes, float confidence_threshold, float *parray,
                                     int max_objects)
{
    float count = predict[0];
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    if (position >= count)
        return;

    float *pitem = predict + 1 + position * (sizeof(Detection) / sizeof(float));
    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float confidence = pitem[4];
    if (confidence < confidence_threshold)
        return;

    float left = pitem[0];
    float top = pitem[1];
    float right = pitem[2];
    float bottom = pitem[3];
    float label = pitem[5];

    float *pout_item = parray + 1 + index * bbox_element;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop,
                                float bright, float bbottom)
{
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float *bboxes, int max_objects, float threshold)
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = bboxes[0];
    if (position >= count)
        return;

    float *pcurrent = bboxes + 1 + position * bbox_element;
    for (int i = 0; i < count; ++i)
    {
        float *pitem = bboxes + 1 + i * bbox_element;
        if (i == position || pcurrent[5] != pitem[5])
            continue;
        if (pitem[4] >= pcurrent[4])
        {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;
            float iou =
                box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);
            if (iou > threshold)
            {
                pcurrent[6] = 0;
                return;
            }
        }
    }
}

void cuda_decode(float *predict, int num_bboxes, float confidence_threshold, float *parray, int max_objects,
                 cudaStream_t stream)
{
    int block = 256;
    int grid = ceil(num_bboxes / (float)block);
    decode_kernel<<<grid, block, 0, stream>>>((float *)predict, num_bboxes, confidence_threshold, parray, max_objects);
}

void cuda_nms(float *parray, float nms_threshold, int max_objects, cudaStream_t stream)
{
    int block = max_objects < 256 ? max_objects : 256;
    int grid = ceil(max_objects / (float)block);
    nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold);
}
