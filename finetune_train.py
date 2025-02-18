import os
import sys

import torch
import monai
import random
import fnmatch
import logging
import argparse
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from model import SwinClassifier
from monai.metrics import ROCAUCMetric
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from monai.data import CacheDataset, DataLoader, decollate_batch, CSVSaver
from monai.transforms import (
    LoadImaged,
    Compose,
    Resized,
    Activations, 
    AsDiscrete,
    ScaleIntensityd,
    EnsureTyped
)

def get_data(path):
    # dataset
    PPS_data_dir = os.path.join(path, 'PPS')
    PD_data_dir = os.path.join(path, 'PD')

    # Get all files
    PPS_data = [os.path.join(root, f) for root, _, files in os.walk(PPS_data_dir) 
                for f in fnmatch.filter(files, '*.nii.gz')]
    
    PD_data = [os.path.join(root, f) for root, _, files in os.walk(PD_data_dir) 
               for f in fnmatch.filter(files, '*.nii.gz')]

    # Shuffle to ensure randomness
    random.shuffle(PPS_data)
    random.shuffle(PD_data)

    # Number of validation samples
    val_PD_size = 200 # Specify your val PD samples
    val_PPS_size = 200  # Specify your val PPS samples

    # Number of training samples
    train_PD_size = 600 # Specify your train PD samples
    train_PPS_size = 600 # Specify your train PPS samples

    # Assign validation data
    val_PD_data = set(PD_data[:val_PD_size])
    val_PPS_data = set(PPS_data[:val_PPS_size])

    # Remaining data after validation
    remaining_PD_data = set(PD_data[val_PD_size:])
    remaining_PPS_data = set(PPS_data[val_PPS_size:])

    # Assign training data
    train_PD_data = set(list(remaining_PD_data)[:train_PD_size])
    train_PPS_data = set(list(remaining_PPS_data)[:train_PPS_size])

    # Remaining data after training (for test set)
    remaining_PD_data -= train_PD_data
    remaining_PPS_data -= train_PPS_data

    # Assign test data
    test_PD_data = set(list(remaining_PD_data)[:200])  # Specify your test PD samples
    test_PPS_data = set(list(remaining_PPS_data)[:200])  # Specify your test PPS samples

    # Convert to list of dictionaries
    train_data = [{"img": img, "label": 0} for img in train_PPS_data] + \
                 [{"img": img, "label": 1} for img in train_PD_data]
    
    val_data = [{"img": img, "label": 0} for img in val_PPS_data] + \
               [{"img": img, "label": 1} for img in val_PD_data]
    
    test_data = [{"img": img, "label": 0} for img in test_PPS_data] + \
                [{"img": img, "label": 1} for img in test_PD_data]

    return train_data, val_data, test_data

def get_features_hook(module, input, output):
    global features
    features = output

def main_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    results_path = '/Data/PD_vs_PPS/results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        
    # save args
    argsDict = args.__dict__
    argsPath = os.path.join(results_path, 'args.txt')
    with open(argsPath, 'w') as f:
        f.writelines('------------------ start --------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ':' + str(value) + '\n')
        f.writelines('------------------ end --------------' + '\n')

    # Define transforms for image
    transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"],spatial_size=(128,128,64)),
            EnsureTyped(keys=["img"])
        ]
    )
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])   

    #Define a classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
    net = SwinClassifier()
    net = net.to(device)    
 
    # Define Hyper-paramters for training loop
    max_epoch = args.epoch
    val_interval = 1
    batch_size = args.batch
    lr = args.lr

    #Define loss & optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=lr)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=max_epoch, eta_min=1e-6)
    auc_metric = ROCAUCMetric()  
   
    train_files, val_files, test_files = get_data(args.dataroot)
    df = pd.DataFrame(train_files)
    df.to_csv(f'{results_path}/train.csv', index=False)
    df_1 = pd.DataFrame(val_files)
    df_1.to_csv(f'{results_path}/validation.csv', index=False)
    df_2 = pd.DataFrame(test_files)
    df_2.to_csv(f'{results_path}/test.csv', index=False)

    # create a training data loader
    train_ds =  CacheDataset(data=train_files, transform=transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    
    # create a validation data loader
    val_ds =  CacheDataset(data=val_files, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
    
    learning_rate = []
    train_loss_values = []
    accuracy_values = []
    auc_values = []

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()
    for epoch in range(max_epoch):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epoch}")
        net.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]
            learning_rate.append(current_lr)           
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        train_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                saver = CSVSaver(output_dir=results_path, filename=f"{net.__class__.__name__}_PD_vs_PPS_predictions.csv")
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, net(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                                        
                    val_outputs = net(val_images).argmax(dim=1)
                    saver.save_batch(val_outputs, val_images.meta)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                #
                accuracy_values.append(acc_metric)
                auc_values.append(auc_result)
                #
                del y_pred_act, y_onehot
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(net.state_dict(), f"{results_path}/best_SwinClassifier.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
                
                plt.figure(1,figsize=(8,8))
                plt.subplot(2,2,1)
                plt.plot(learning_rate)
                plt.xlabel('Epoches')
                plt.ylabel('Learning_rate')
                plt.xlim(0,max_epoch)
                plt.title('CosineAnnealingLR')
                
                plt.subplot(2,2,2)
                plt.plot(train_loss_values)
                plt.xlim(0,max_epoch)
                plt.grid()
                plt.title('Training loss')

                plt.subplot(2, 2, 3)  
                plt.plot(accuracy_values)
                plt.xlim(0,max_epoch)
                plt.grid()
                plt.title('Accuracy')
                
                plt.subplot(2, 2, 4)  
                plt.plot(auc_values)
                plt.xlim(0,max_epoch)
                plt.grid()
                plt.title('AUC')
                
                plt.tight_layout()
                plt.suptitle('PPS_vs_PD')
                plt.savefig(os.path.join(results_path, f"{args.task}_loss_4f.png"))
                plt.close(1)
        
    df = pd.DataFrame()
    metrics = {'best_AUC':[auc_result], 'best_accuracy':[best_metric], 'best_epoch':[best_metric_epoch],
               'epoch':[args.epoch], 'batch':[args.batch], 'task':[args.task]}
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(results_path, f"{args.task}_4f_train_results.csv"), index=False)    

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    saver.finalize()
    writer.close()    

def main():
    parser = argparse.ArgumentParser(description='training parameters')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='0', type=str, help='gpu id')
    parser.add_argument('--task', default='PPS_vs_PD', help='choose task')
    parser.add_argument('--dataroot', default='/Data/PD/data')   
    parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='entire epoch number')
    parser.add_argument('--batch', default=4, type=int, help='batch size')
    
    args = parser.parse_args()
    main_worker(args)
    
if __name__=="__main__":
    main()
