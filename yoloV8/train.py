

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='yoloV8.yaml', 
    batch=8, 
    epochs=2, 
    imgsz=640, 
    save_period=1000,
    #project="/workspace/yoloV8/runs/detect",
    )
