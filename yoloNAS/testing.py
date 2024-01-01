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


    checkpoint_dir = '../checkpoints'
    model_path = "../inference_model" + "/" + os.listdir("../inference_model")[0]
    input_folder = "../data/test/images"
    #output_folder = "../inference_output"
    MODEL_ARCH = "yolo_nas_l"
    classes = ['Deer', 'Roe Deer', 'Chamois', 'Wild Boar', 'Rabbit', 'Horse', 'Sika Deer', 'Buffalo','Sheep' ]




    trainer = Trainer(
    experiment_name=MODEL_ARCH,
    ckpt_root_dir=checkpoint_dir    
    )


    model = get_model(MODEL_ARCH, num_classes=len(classes), checkpoint_path=model_path)
   
    image_files = glob.glob(os.path.join(input_folder, '*.png'))
    test_data_loader 
    test_results = trainer.test(model=model, test_loader=test_data_loader, test_metrics_list=test_metrics)








if __name__ == "__main__":
    main()