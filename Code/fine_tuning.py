# from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets
from transformations import get_transformer
import time
import copy
import os
import pandas as pd
from PIL import ImageFile
from fine_tuning_config_file import *
from model_loaders import get_model
from numpy import random
from sklearn.metrics import confusion_matrix, roc_auc_score
from overlooked_images import generate_falsepositive_list
from DatasetGenerator import load_dataset

# Code mostly copied from https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial/blob/master/main_fine_tuning.py
# this file

def compute_measures(epoch, phase, dset_sizes, running_loss, folder, preds_list, labels_list):
    tn, fp, fn, tp = confusion_matrix(labels_list, preds_list).ravel()
    auc = roc_auc_score(labels_list, preds_list)
    epoch_loss = running_loss / dset_sizes[phase]
    epoch_acc = (tp + tn) / (tn + tp + fp + fn)
    fields = ["epoch", "phase", "epoch_loss", "tn", "fp", "fn", "tp", "auc"]
    new_row = [epoch, phase, epoch_loss, tn, fp, fn, tp, auc]
    log_path = os.path.join(folder, "metrics.csv")
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
        log.loc[len(log)] = new_row
    else:
        log = pd.DataFrame({fields[i]: new_row[i] for i in range(len(fields))}, index=[0])

    log.to_csv(log_path, index=False)
    return epoch_loss, epoch_acc


if __name__ == '__main__':

    data_transforms = get_transformer("normalize_256")

    payload = {"task": "bikelane", "phase": "train", "transform": data_transforms, "oversamplingrate": 10, "split": 0.2}

    dsets = {'train': load_dataset(**payload, set="train"),
             'val':   load_dataset(**payload, set="val")}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                                   shuffle=True, num_workers=12)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}

    def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=50):
        since = time.time()
        best_loss_epoch = 1
        best_model = model
        best_loss = 100
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    optimizer = lr_scheduler(optimizer, epoch)
                    model.train()  # Set model to training mode
                else:
                    model.eval()

                # at the start of  each epoch, reset metric counts
                running_loss = 0.0
                #c_matrix = 0
                preds_list=[]
                labels_list=[]

                # Iterate over data.
                for data in dset_loaders[phase]:
                    inputs = data["Image"]
                    labels = data["Label"]
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()

                    # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()


                    #c_matrix += confusion_matrix(labels.cpu(), preds.cpu())

                    preds_list=preds_list + list(preds.cpu())
                    labels_list=labels_list + list(labels.cpu())
                epoch_loss, epoch_acc = compute_measures(epoch=epoch, phase=phase, dset_sizes=dset_sizes,
                                                         running_loss=running_loss, folder=folder,
                                                         preds_list=preds_list, labels_list=labels_list)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val':
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        print('new best loss is = ', best_loss)
                        best_loss_epoch = epoch

                    if True:
                        best_model = copy.deepcopy(model)
                    print(f"Best loss was found in epoch {best_loss_epoch}")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_loss))
        print('returning and looping back')
        return best_model


    # This function changes the learning rate over the training model.
    def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
        """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
        lr = init_lr * (DECAY_WEIGHT ** (epoch // lr_decay_epoch))

        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer


    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    experiment_name = EXPERMIMENT_NAME
    # define destination folder
    folder = "../Results/" + experiment_name

    if not os.path.exists(os.path.split(folder)[0]):
        os.mkdir(os.path.split(folder)[0])
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        folder_changed=folder
        i = 2
        while os.path.exists(folder_changed):
            folder_changed = folder + "_" + str(i)
            i+=1
        folder = folder_changed
        os.mkdir(folder)

    model_ft = get_model("resnet", pretrained=True)
    # loss adaptation for imbalanced learning
    weights = [1, 1]  # as class distribution. 1879 negativs, 110 positives.
    if torch.cuda.is_available():
        class_weights = torch.FloatTensor(weights).cuda()
    else:
        class_weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.001)
    if torch.cuda.is_available():
        criterion.cuda()
        model_ft.cuda()

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=50)

    # Save model
    torch.save(model_ft.state_dict(), folder + "/fine_tuned_best_model.pt")

    # save the false positives
    generate_falsepositive_list("bikelane", folder, model_ft)
