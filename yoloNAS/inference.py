import os
import cv2
import argparse
from tqdm import tqdm
from super_gradients.training.models import get as get_model
from super_gradients import init_trainer
from super_gradients import setup_device
from os import listdir
import torch
import glob
import time


def main():

    init_trainer()

    # parser = argparse.ArgumentParser(description="Inference script for object detection")
    # parser.add_argument("-i", "--input_folder", help="Path to the folder containing input images")
    # parser.add_argument("-o", "--output_folder", help="Path to save output images with bounding boxes")
    # args = parser.parse_args()

    size = "l"
    dataset = "all"

    model_path = f'/workspace/YAD/results/{dataset}/nas_{size}_1/ckpt_best.pth'

    input_folder = "/workspace/YAD/data/test/images"
 
    output_folder = f"../inference_output/nas/{size}"
    #output_folder = f"../inference_output/nas/test"

    MODEL_ARCH = f"yolo_nas_{size}"
    classes = ['Deer', 'Fallow Deer', 'Horse', 'Rabbit', 'Roe Deer', 'Wild Boar']


    # Load the trained model
    model = get_model(MODEL_ARCH, num_classes=len(classes), checkpoint_path=model_path)
    model.eval()

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    timer = time.time()
    # images_predictions = model.predict(image_files, iou=0.5, conf=0.7)

    # images_predictions.show(box_thickness=2, show_confidence=True)
    # images_predictions.save("testname", output_folder=output_folder)

    print("Start prediction")
    for image in image_files:
        pred = model.predict(image, iou=0.5, conf=0.7)
        pred.show(box_thickness=2, show_confidence=True)
        pred.save(output_folder=output_folder)
        os.rename(os.path.join(output_folder, "pred_0.jpg"), os.path.join(output_folder, os.path.basename(image)))
    print("Prediction done!")

    print(f"Time taken for inference: {time.time() - timer} seconds")
    num_images = len(image_files)
    print(f"ms per image: {(time.time() - timer) * 1000 / num_images}")

if __name__ == "__main__":
    main()