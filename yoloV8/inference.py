import ultralytics
from ultralytics import YOLO
import subprocess

model_used = "s"

ultralytics.checks()
model = YOLO(f'../results/v8_{model_used}_1/weights/best.pt')

model.val(data='configs/dataDeerTest.yaml')
model.predict(source='../dataDeer/test/images', save=True, save_txt=True, save_conf=True , conf=0.7, line_width=1) #imgsz=640

src_inf= "/usr/src/ultralytics/runs/detect/predict"
dest_inf = f"/workspace/inference_output/v8/{model_used}"
cmd_inf = f"mv {src_inf} {dest_inf}"

src_val = "/usr/src/ultralytics/runs/detect/val"
dest_val = f"/workspace/inference_output/v8/{model_used}"
cmd_val = f"mv {src_val} {dest_val}"

subprocess.run(cmd_inf, shell=True)
subprocess.run(cmd_val, shell=True)
