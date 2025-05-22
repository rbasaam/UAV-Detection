import torch
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights='models/transformer.pth')
model.export()
print("Model exported to ONNX format.")
