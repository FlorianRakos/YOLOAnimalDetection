import ultralytics
ultralytics.checks()

from ultralytics import YOLO

model = YOLO('runs/yolov8l/train/weights/best.pt')

model.predict(source='../data/test/images', save=True , conf=0.7, line_width=1) #imgsz=640


# from ultralytics import YOLO
# import cv2
# import os


# results = model('https://ultralytics.com/images/bus.jpg')
# img = cv2.imread('bus.jpg')

# path = "another_folder"
# if not os.path.exists(path):
#     os.mkdir(path)

# for result in results:
#     boxes = result.boxes.cpu().numpy()
#     for i, box in enumerate(boxes):
#         r = box.xyxy[0].astype(int)
#         crop = img[r[1]:r[3], r[0]:r[2]]
#         filename = str(i) + ".jpg"
#         if (box.conf[0] > 0.5):
#             filename = os.path.join(path, filename)
#         cv2.imwrite(filename, crop)