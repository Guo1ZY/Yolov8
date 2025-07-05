# 验证程序
# 此处给出一些常用的参数，其他参数见官方文档或教程修改

from ultralytics import YOLO

# 训练好的pt文件路径
data_path = (
    "/home/zzzing/桌面/代码/RC2024/Yolov8_With_Python/Cpp_Samples/py/yolov8/model/best.pt"
)

# data目录下yaml路径
yaml_path = "/home/zzzing/桌面/代码/RC2024/Yolov8_With_Python/Train/data/silver.yaml"

# batch大小，根据显存大小尽量向上调整
batch_size = 32

# 图片压缩后的大小
# 与训练一致
image_size = (256, 256)

# 选择设备，可选cpu、0、1等
device = 0

# 进行验证
if __name__ == "__main__":
    model = YOLO(data_path)

    metrics = model.val(
        data=yaml_path, imgsz=image_size, batch=batch_size, device=device
    )
