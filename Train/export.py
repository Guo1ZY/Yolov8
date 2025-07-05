# 导出程序
# 此处给出一些常用的参数，其他参数见官方文档或教程修改

from ultralytics import YOLO

# 训练好的pt文件路径
data_path = "/media/zy/9361-7A75/img/img01/runs/train2/ball/weights/best.pt"

# opset版本
opset = 12

# 图片压缩后的大小
# 与训练一致
image_size = (512, 512)

# FP16 quantization
half = True

# dynamic axes
dynamic = False

# 开始导出
if __name__ == "__main__":
    model = YOLO(data_path)

    model.export(
        format="onnx", opset=opset, imgsz=image_size, half=half, dynamic=dynamic
    )
