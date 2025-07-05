# 训练程序
# 此处给出一些常用的参数，其他参数见官方文档或教程修改

from ultralytics import YOLO

# 加载的预训练权重，如果是空则默认选择yolov8n.pt
data_path = (
    "/media/zy/9361-7A75/img/img01/yolov8n.pt"
)

# 生成训练文件的路径
# 工程路径
project = "/media/zy/9361-7A75/img/img01/runs/train2/"
# 名称
name = "ball"

# Save checkpoint every x epochs (disabled if < 1)
save_period = 1000

# 是否从上次打断的节点继续训练
resume = False

# data目录下yaml路径
yaml_path = "/media/zy/9361-7A75/img/img01/ball.yaml"

# batch大小，根据显存大小尽量向上调整
# batch越大，收敛越快越稳定
batch_size = 64

# 训练轮数，如果数据集比较小可以设置大一些，保证训练loss降到0.1以下
epochs = 100

# 图片压缩后的大小
# 对于需要检测小的物体，可以调大该Size，但是会占用更多显存，同时也会导致推理时间更长
# 需要是32的倍数
image_size = (256, 256)

# 选择训练的设备，可选cpu、0、1等
device = 0

# 多线程训练
workers = 8

# 进行训练
if __name__ == "__main__":
    # 加载模型
    if data_path is None:
        data_path = "yolov8n.pt"

    model = YOLO(data_path)

    # 开始训练
    results = model.train(
        model=data_path,
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        workers=workers,
        resume=resume,
        save_period=save_period,
        project=project,
        name=name,
    )
