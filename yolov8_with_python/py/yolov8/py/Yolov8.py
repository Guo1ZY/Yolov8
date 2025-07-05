# 使用ultralytics封装的函数进行推理

# 根据源文件yolov8/ultralytics/ultralytics/yolo/utils/__init__.py
# 设置环境变量YOLO_VERBOSE = False即可关闭原推理输出
# 如要查看推理输出可以设置环境变量YOLO_VERBOSE = True
import os

os.environ["YOLO_VERBOSE"] = str(False)

from ultralytics import YOLO


class Yolov8:
    def __init__(self, modelPath, confThreshold=0.5, cudaEnabled=False):
        self.modelPath = modelPath  # 模型路径
        self.confThreshold = confThreshold  # 置信度
        self.cudaEnabled = cudaEnabled  # 是否使用cuda

        # 加载模型
        self.model = YOLO(modelPath)
        if self.cudaEnabled:
            self.model.cuda()

        # print(self.model)

    def detect(self, image):
        print("hello0")

        # 检测
        results = self.model(image)

        print("hello1")

        return_value = []
        for result in results:
            # 下载到cpu
            result = result.cpu()

            xywhs = result.boxes.xywh.numpy()  # 框
            confs = result.boxes.conf.numpy()  # 置信度
            clss = result.boxes.cls.numpy()  # 类别

            for xywh, conf, cls in zip(xywhs, confs, clss):
                # 过滤置信度低的目标
                if conf < self.confThreshold:
                    continue

                x, y, w, h = xywh
                return_value.append([x, y, w, h, conf, cls])

        return return_value
