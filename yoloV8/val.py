import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO('runs/detect/train21/weights/best.pt')

model.val("animal.yaml", imgsz=640)
