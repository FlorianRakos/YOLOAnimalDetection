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
from super_gradients.training import Trainer, models, dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

def main():

    init_trainer()

    size = "s"
    dataset = 'data'

    checkpoint_dir = '../checkpoints'
    model_path = f'/workspace/YAD/results/all/nas_{size}_1/ckpt_best.pth'
    input_folder = "../data/test/images"
    #output_folder = "../inference_output"
    MODEL_ARCH = f"yolo_nas_{size}"
    classes = ['Deer', 'Fallow Deer', 'Horse', 'Rabbit', 'Roe Deer', 'Wild Boar']


    trainer = Trainer(
    experiment_name=MODEL_ARCH,
    ckpt_root_dir=checkpoint_dir    
    )


    model = get_model(MODEL_ARCH, num_classes=len(classes), checkpoint_path=model_path)
   
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    BATCH_SIZE = 32
    WORKERS = 8
    ROOT_DIR = ''
    train_imgs_dir = f'../{dataset}/train/images'
    train_labels_dir = f'../{dataset}/train/labels'
    val_imgs_dir = f'../{dataset}/valid/images'
    val_labels_dir = f'../{dataset}/valid/labels'
    checkpoint_dir = 'runs'
    test_imgs_dir = f'../{dataset}/test/images'
    test_labels_dir = f'../{dataset}/test/labels'

    dataset_params = {
        'data_dir':ROOT_DIR,
        'train_images_dir':train_imgs_dir,
        'train_labels_dir':train_labels_dir,
        'val_images_dir':val_imgs_dir,
        'val_labels_dir':val_labels_dir,
        'test_images_dir':test_imgs_dir,
        'test_labels_dir':test_labels_dir,
        'classes':classes
    }

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['test_images_dir'],
            'labels_dir': dataset_params['test_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':BATCH_SIZE,
            'num_workers':WORKERS
        }
    )

    test_metrics = [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        ),
        DetectionMetrics_050_095(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ]

    test_results = trainer.test(model=model, test_loader=test_data, test_metrics_list=test_metrics)


if __name__ == "__main__":
    main()