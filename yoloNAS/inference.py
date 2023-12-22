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


def main():

    init_trainer()

    # parser = argparse.ArgumentParser(description="Inference script for object detection")
    # parser.add_argument("-i", "--input_folder", help="Path to the folder containing input images")
    # parser.add_argument("-o", "--output_folder", help="Path to save output images with bounding boxes")
    # args = parser.parse_args()


    model_path = "../inference_model" + "/" + os.listdir("../inference_model")[0]
    input_folder = "../data/test/images"
    output_folder = "../inference_output"
    MODEL_ARCH = "yolo_nas_l"
    classes = ['Deer', 'Roe Deer', 'Chamois', 'Wild Boar', 'Rabbit', 'Horse', 'Sika Deer', 'Buffalo','Sheep' ]


    # Load the trained model
    model = get_model(MODEL_ARCH, num_classes=len(classes), checkpoint_path=model_path)
    model.eval()
   
    image_files = glob.glob(os.path.join(input_folder, '*.png'))
    images_predictions = model.predict(image_files, iou=0.5, conf=0.7)
    images_predictions.show(box_thickness=2, show_confidence=True)
    images_predictions.save(output_folder=output_folder)


if __name__ == "__main__":
    main()