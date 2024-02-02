from super_gradients import init_trainer
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models, dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.utils.distributed_training_utils import setup_gpu
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from tqdm.auto import tqdm
import os
import requests
import zipfile
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import argparse
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


parser = argparse.ArgumentParser(description="Train a model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-e", "--epochs",  help="add number of epochs")
parser.add_argument("-m", "--model",  help="add model architecture")
args = parser.parse_args()
config = vars(args)


if (args.model == 'm'):
    MODEL_ARCH = 'yolo_nas_m'
    WORKERS = 8
elif (args.model == 'l'):
    MODEL_ARCH = 'yolo_nas_l'
    WORKERS = 6
else:
    MODEL_ARCH = 'yolo_nas_s'
    WORKERS = 12

dataset = 'dataDeer'



init_trainer()
setup_gpu(num_gpus=1)

ROOT_DIR = ''
train_imgs_dir = f'../{dataset}/train/images'
train_labels_dir = f'../{dataset}/train/labels'
val_imgs_dir = f'../{dataset}/valid/images'
val_labels_dir = f'../{dataset}/valid/labels'
checkpoint_dir = 'runs'
test_imgs_dir = f'../{dataset}/test/images'
test_labels_dir = f'../{dataset}/test/labels'
classes = ['Deer', 'Roe Deer' ] #, 'Chamois', 'Wild Boar', 'Rabbit', 'Horse', 'Sika Deer', 'Buffalo','Sheep'


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

# Global parameters.
EPOCHS = int(args.epochs)
BATCH_SIZE = 32


train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':BATCH_SIZE,
        'num_workers':WORKERS
    }
)

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


train_params = {
    'silent_mode': False,   #controls whether the training process will display information and progress updates
    "average_best_models":True,   #average the parameters of the best models
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 5e-7, #1e-6   # increasing the learning rate over a specified number of epochs
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-5, # 5e-4,
    "lr_mode": "cosine",   #learning rate follows a cosine function's curve throughout the training process
    "cosine_final_lr_ratio": 0.05, #0.1  #ratio of the final learning rate to the initial learning rate
        
    # "initial_lr": 0.1,
    # "lr_mode":"StepLRScheduler",
    # "lr_updates": [100, 150, 200],
    # "lr_decay_factor": 0.1,
    # "lr_mode": "StepLRScheduler",
    # "step_lr_update_freq": 2.4,
    # "initial_lr": 0.016,
    # "lr_warmup_epochs": 3,
    # "warmup_initial_lr": 1e-6,
    # "lr_decay_factor": 0.97,

    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},   #prevent overfitting by penalizing large weights
    "zero_weight_decay_on_bias_and_bn": True,   #True: zero weight decay on bias and batch normalization parameters
    "ema": True,   #Exponential Moving Average
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": EPOCHS,
    "mixed_precision": True,   #True: combination of 16-bit and 32-bit floating-point arithmetic to speed up training while conserving memory
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
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
    ],
    "metric_to_watch": 'mAP@0.50:0.95'
}

#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

trainer = Trainer(
    experiment_name=MODEL_ARCH,
    ckpt_root_dir=checkpoint_dir
)

model = models.get(
    MODEL_ARCH,
    num_classes=len(dataset_params['classes']),
    pretrained_weights="coco") #.to(DEVICE)


trainer.train(
    model=model,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data,
    test_loaders={"TestSet": test_data}
)
