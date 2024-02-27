
import ultralytics
ultralytics.checks()
from ultralytics import YOLO


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model size", default="n")
parser.add_argument("--epochs", help="number of epochs", default=300)
args = parser.parse_args()


if args.model == "n":
    model_name = "yolov8n"
elif args.model == "s":
    model_name = "yolov8s"
elif args.model == "m":
    model_name = "yolov8m"
elif args.model == "l":
    model_name = "yolov8l"
else:
    print("Invalid model size")
    exit()


model = YOLO(model_name + '.pt')

results = model.train(
    data='configs/data.yaml',
    cfg='configs/no_aug.yaml', 
    batch=8, 
    epochs=int(args.epochs),
    imgsz=640, 
    save_period=1000,
    project="/workspace/yoloV8/runs/" + model_name,
    )
