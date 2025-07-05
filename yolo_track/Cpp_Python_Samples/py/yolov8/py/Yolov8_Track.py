# 设置环境变量YOLO_VERBOSE = False即可关闭原推理输出
# 如要查看推理输出可以设置环境变量YOLO_VERBOSE = True
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

np.seterr(all="raise")
os.environ["YOLO_VERBOSE"] = str(False)

from ultralytics import YOLO


class Yolov8_Track:
    def __init__(self, modelPath, confThreshold=0.6, cudaEnabled=True):
        self.modelPath = modelPath  # 模型路径
        self.confThreshold = confThreshold  # 置信度
        self.cudaEnabled = cudaEnabled  # 是否使用cuda

        self.inputShape = [256, 256]

        # 加载模型
        self.model = YOLO(modelPath)
        if self.cudaEnabled:
            self.model.cuda()

        # print(self.model)

    def track_t(self, image):
        result = []

        # 检测
        if self.cudaEnabled:
            # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
            print("running on cuda")
            results = self.model.track(
                image,
                persist=True,
                imgsz=self.inputShape,
                device=0,
                tracker="bytetrack.yaml",
            )
            print("results:", results)
        else:
            print("running on cpu")
            results = self.model.track(
                image,
                persist=True,
                imgsz=self.inputShape,
                device="cpu",
                tracker="bytetrack.yaml",
            )

        return_value = []
        for result in results:
            # 下载到cpu
            result = result.cpu()

            xywhs = result.boxes.xywh.numpy()  # 框
            confs = result.boxes.conf.numpy()  # 置信度
            clss = result.boxes.cls.numpy()  # 类别
            ids = result.boxes.id.numpy()  # 追踪id

            for xywh, conf, cls, id in zip(xywhs, confs, clss, ids):
                # 过滤置信度低的目标
                if conf < self.confThreshold:
                    continue
                x, y, w, h = xywh
                return_value.append([x, y, w, h, conf, cls, id])
            # print("return_value:", return_value)
        return return_value
