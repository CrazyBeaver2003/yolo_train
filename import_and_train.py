from ultralytics import YOLO
import torch
print(f'Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "CPU"})')

# This will automatically download yolov11n.pt if not present
model = YOLO('yolo11n.pt')

results = model.train(
    data='photos/dataset.yaml',
    epochs=100,
    imgsz=640,
    device='gpu',
    batch=16,
    patience=20,
    save=True,
    plots=True,
)

model.export(format='onnx')
