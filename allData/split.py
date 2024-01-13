
# train: 0.8, val: 0.1, test: 0.1

import os
import shutil
import random

path = os.getcwd()
images = os.listdir(path + '/images')
labels = os.listdir(path + '/labels')

print("Images: " , len(images))
print("Labels: " , len(labels))

# create folders
os.mkdir(path + '/train')
os.mkdir(path + '/val')
os.mkdir(path + '/test')

# create subfolders
os.mkdir(path + '/train/images')
os.mkdir(path + '/train/labels')
os.mkdir(path + '/val/images')
os.mkdir(path + '/val/labels')
os.mkdir(path + '/test/images')
os.mkdir(path + '/test/labels')

# split images
random.shuffle(images)
train_images = images[:int(0.8 * len(images))]
val_images = images[int(0.8 * len(images)):int(0.9 * len(images))]
test_images = images[int(0.9 * len(images)):]

# split labels
train_labels = []
val_labels = []
test_labels = []

for image in train_images:
    train_labels.append(image[:-4] + '.txt')
for image in val_images:
    val_labels.append(image[:-4] + '.txt')
for image in test_images:
    test_labels.append(image[:-4] + '.txt')


# copy images and labels to subfolders
# for image in train_images:
#     shutil.copy(path + '/images/' + image, path + '/train/images/' + image)
# for image in val_images:
#     shutil.copy(path + '/images/' + image, path + '/val/images/' + image)
# for image in test_images:
#     shutil.copy(path + '/images/' + image, path + '/test/images/' + image)


# for label in train_labels:
#     shutil.copy(path + '/labels/' + label, path + '/train/labels/' + label)
# for label in val_labels:
#     shutil.copy(path + '/labels/' + label, path + '/val/labels/' + label)
# for label in test_labels:
#     shutil.copy(path + '/labels/' + label, path + '/test/labels/' + label)

copyCounter = 0

def copy_files(file_list, source_path, destination_path):
    for file in file_list:
        source_file_path = os.path.join(source_path, file)
        destination_file_path = os.path.join(destination_path, file)
        try:
            shutil.copy(source_file_path, destination_file_path)
            global copyCounter
            copyCounter += 1
        except FileNotFoundError as e:
            print(f"File {file} not found: {e}")

# Copy images
copy_files(train_images, path + '/images/', path + '/train/images/')
copy_files(val_images, path + '/images/', path + '/val/images/')
copy_files(test_images, path + '/images/', path + '/test/images/')

# Copy labels
copy_files(train_labels, path + '/labels/', path + '/train/labels/')
copy_files(val_labels, path + '/labels/', path + '/val/labels/')
copy_files(test_labels, path + '/labels/', path + '/test/labels/')

print("Copied " , copyCounter , " files") #