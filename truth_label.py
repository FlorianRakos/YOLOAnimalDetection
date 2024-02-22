import os
import cv2
import glob

cwd = os.getcwd()

images = os.path.join(cwd, 'data/test/images')
labels = os.path.join(cwd, 'data/test/labels')
output_folder = os.path.join(cwd, 'inference_output/truth')

image_files = glob.glob(os.path.join(images, '*.jpg'))
label_files = glob.glob(os.path.join(labels, '*.txt'))

image_files.sort()
label_files.sort()

print("Images: " , len(image_files))
print("Labels: " , len(label_files))

classes = ['Deer', 'Fallow Deer', 'Horse', 'Rabbit', 'Roe Deer', 'Wild Boar']

for image, label in zip(image_files, label_files):
    print("Processing Image: ", os.path.basename(image), " --- Label: ", os.path.basename(label))
    img = cv2.imread(image)
    with open(label, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            class_id = int(line[0])
            x = int(float(line[1]) * img.shape[1])
            y = int(float(line[2]) * img.shape[0])
            w = int(float(line[3]) * img.shape[1])
            h = int(float(line[4]) * img.shape[0])

            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)
            cv2.putText(img, classes[class_id], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imwrite(os.path.join(output_folder, os.path.basename(image)), img)


