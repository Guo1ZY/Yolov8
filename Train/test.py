# 测试程序
# 此处给出一些常用的参数，其他参数见官方文档或教程修改

from ultralytics import YOLO

# 训练好的pt文件路径
data_path = (
    "/home/zzzing/桌面/代码/RC2024/Yolov8_With_Python/Cpp_Samples/py/yolov8/model/best.pt"
)

# 图片集路径
image_path = "/home/zzzing/桌面/3_22_gongye/jpgs"

# 置信度
conf = 0.5

# 图片压缩后的大小
# 与训练一致
image_size = (256, 256)

# 选择设备，可选cpu、0、1等
device = 0

# 开始测试
if __name__ == "__main__":
    model = YOLO(data_path)

    model.predict(
        source=image_path, conf=conf, imgsz=image_size, device=device, save=True
    )
