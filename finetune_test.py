#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:07:36 2023

@author: gn3
"""
import logging
import os
import sys
import time
import csv
import argparse
import fnmatch
import pandas as pd
import matplotlib.colors
import matplotlib
import numpy as np
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.networks.nets import DenseNet121, resnet50, SENet154, SEResNet101, SEResNeXt101, resnet18
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.data import CSVSaver
from model import ViTClassifier, SwinClassifier
from monai.data import decollate_batch, CacheDataset, DataLoader
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, Resized, ScaleIntensityd
warnings.filterwarnings('ignore')

def get_data(path, csv_file_path):
    # dataset
    PPS_data_dir = os.path.join(path,'PPS')
    PD_data_dir = os.path.join(path,'PD')
    
    PPS_data = []    
    for path, dirs, files in os.walk(PPS_data_dir):
        for f in fnmatch.filter(files, '*.nii.gz'):
            PPS_data.append(os.path.join(path, f))             
    
    PD_data = []
    for path, dirs, files in os.walk(PD_data_dir):
        for f in fnmatch.filter(files, '*.nii.gz'):
            PD_data.append(os.path.join(path, f)) 
    
    random.shuffle(PPS_data)
    random.shuffle(PD_data)
    
    csv_file = csv_file_path
    id_list = []

    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            id_list.append(row['ID'])
            
    PPS_files = [{"img": img, "label": 0} for img in PPS_data if any(id_part in img for id_part in id_list)]
    PD_files = [{"img": img, "label": 1} for img in PD_data if any(id_part in img for id_part in id_list)]       
    
    data = PPS_files + PD_files

    return data


def test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
   
    data_path = '/Data/PD/data'
    test_results_path = '/Data/PD_vs_PPS/test_results'

    # get data
    test_file_path = '/Data/PD_vs_PPS/results/test.csv'

    test_files = get_data(data_path, test_file_path)
    df = pd.DataFrame(test_files)
    df.to_csv(f'{test_results_path}/test_files.csv', index=False)
    print('Total Number of Test Data Samples: {}'.format(len(test_files)))      
    print('-' * 50)

    # Define transforms for image
    transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(128, 128, 64))
        ]
    )
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    
    # create a training data loader
    test_ds = CacheDataset(data=test_files, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=4, pin_memory=torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model_path = '/Data/PD_vs_PPS/results/logs'
    
    model1 = resnet50(spatial_dims=3, n_input_channels=3, num_classes=2).to(device)
    model1.load_state_dict(torch.load(os.path.join(model_path, "resnet50", "best_resnet50.pth")))

    model2 = DenseNet121(spatial_dims=3, in_channels=3, out_channels=2).to(device)
    model2.load_state_dict(torch.load(os.path.join(model_path, "DenseNet121", "best_DenseNet121.pth")))

    model3 = ViTClassifier().to(device)
    model3.load_state_dict(torch.load(os.path.join(model_path, "ViT", "best_ViTClassifier.pth")))

    model4 = SwinClassifier().to(device)
    model4.load_state_dict(torch.load(os.path.join(model_path, "SwinUNETR", "best_SwinClassifier.pth")))

    classifiers = [model1, model2, model3, model4]
    results = {}
  
    for model in classifiers:
        model.eval()
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            saver = CSVSaver(output_dir=test_results_path, filename=f"{model.__class__.__name__}_PD_vs_PPS_predictions.csv")
            for test_data in test_loader:
                test_images, test_labels = test_data["img"].to(device), test_data["label"].to(device)
                y = torch.cat([y, test_labels], dim=0)
                y_pred = torch.cat([y_pred, model(test_images)], dim=0)

                test_outputs = model(test_images).argmax(dim=1)
                value = torch.eq(test_outputs, test_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
                saver.save_batch(test_outputs, test_data["img"].meta)
            metric = num_correct / metric_count #accuracy
            print(metric)
            
            labels = [post_label(i) for i in decollate_batch(y, detach=False)]           
            labels_array = np.array([tensor.data.detach().cpu().numpy() for tensor in labels])
            
            # ------------------Cauclate F1 score\recall\precise---------------
            true_label = labels_array[::,1]
            pred_label = y_pred.argmax(dim=1)
            pred_label = np.array([tensor.data.detach().cpu().numpy() for tensor in pred_label])
  
            f1 = f1_score(true_label, pred_label)
            recall = recall_score(true_label, pred_label)
            precision = precision_score(true_label, pred_label, average='binary')            
            
            results[model.__class__.__name__] = {'Accuracy': metric, 'F1': f1, 'Precision': precision, 'Recall': recall}
            print(f"{model.__class__.__name__} evaluation metrics: acc={metric}, f1={f1}, precision={precision}, recall={recall}")
            saver.finalize()
            
    with open(f'{test_results_path}/PD_vs_PPS_test_metrics.csv', 'w', newline='') as csv_file:
        fieldnames = ['metrics'] + list(results.keys())
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fieldnames)
        for metric_name in ['Accuracy', 'F1', 'Precision', 'Recall']:
            row_values = [metric_name] + [results[model_name][metric_name] for model_name in results]
            csv_writer.writerow(row_values)

        
def main():  
    parser = argparse.ArgumentParser(description='test parameters')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='0', type=str, help='gpu id')
    parser.add_argument('--batch', default=6, type=int, help='batch size')
    args = parser.parse_args()
    test(args)

if __name__ == "__main__":
    main()
