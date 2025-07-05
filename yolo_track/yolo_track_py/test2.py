import cv2
import numpy as np
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('/home/zy/桌面/yolo_track/Cpp_Python_Samples/py/yolov8/model/ball/best.pt')

# 打开视频文件
# video_path = "test_track.mp4"
cap = cv2.VideoCapture(0)

# 循环遍历视频帧
while cap.isOpened():
    # 从视频读取一帧
    success, frame = cap.read()

    if success:
        # 取左半张图片
        height, width, _ = frame.shape
        left_half_frame = frame[:, :width // 2, :]

        # 在左半张图片上运行YOLOv8追踪，持续追踪帧间的物体
        results = model.track(left_half_frame, persist=True)
        # 输出每次追踪推理结果的boxes，这些参数实际上是和模型直接predict类似的。
        print(results[0].boxes)
        
        # 在左半张图片上展示结果
        annotated_frame = results[0].plot()
        
        # 创建一个全黑图像，将左半张图片复制到左侧
        full_annotated_frame = np.zeros_like(frame)
        full_annotated_frame[:, :width // 2, :] = annotated_frame

        # 展示带注释的帧
        cv2.imshow("YOLOv8 Tracking", full_annotated_frame)

        # 如果按下'q'则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则退出循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
