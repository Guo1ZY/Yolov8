import sys  # noqa: F401
import argparse
import os
import struct
import torch
import warnings



# .pt文件路径
pt_file = "/home/zy/Yolov11_Train/output_v11/weights/best.pt"
# wts文件路径
wts_file = "./best.wts"

m_type = "detect"

print(f"Generating .wts for {m_type} model")

# Load model
print(f"Loading {pt_file}")

# Initialize
device = "cuda"

# Load model
model = torch.load(pt_file, map_location=device)["model"].float()  # load to FP32

if m_type in ["detect", "seg", "pose"]:
    anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]

    delattr(model.model[-1], "anchors")

model.to(device).eval()

with open(wts_file, "w") as f:
    f.write("{}\n".format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {} ".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
