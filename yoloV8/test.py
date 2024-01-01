import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO('runs/detect/train21/weights/best.pt')

model.predict(source='datasets/video', save=True, imgsz=640, conf=0.5, line_width=1)
