import ultralytics
from ultralytics import YOLO
import subprocess

model_used = "m"
dataset = "all"
datadir = "data"


ultralytics.checks()
model = YOLO(f'../results/{dataset}/v8_{model_used}_1/weights/best.pt')

model.val(data=f'configs/dataTest.yaml')
model.predict(source=f'../{datadir}/test/images', save=True, save_txt=True, save_conf=True , conf=0.7, line_width=1) #imgsz=640



src_inf= "/usr/src/ultralytics/runs/detect/predict"
src_val = "/usr/src/ultralytics/runs/detect/val"
dest = f"/workspace/inference_output/v8/{model_used}"

cmd_inf = f"mv {src_inf} {dest}"
cmd_val = f"mv {src_val} {dest}"
cmd_chmod = f"chmod 777 -R {dest}"

print("ExecuteCMD: ", cmd_inf)
subprocess.run(cmd_inf, shell=True)
print("ExecuteCMD: ", cmd_val)
subprocess.run(cmd_val, shell=True)
print("ExecuteCMD: ", cmd_chmod)
subprocess.run(cmd_chmod, shell=True)
