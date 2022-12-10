# from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
import time
import copy
import os
import pandas as pd
from PIL import ImageFile
from fine_tuning_config_file import *
from model_loaders import get_model
from numpy import random
from sklearn.metrics import confusion_matrix
# Code mostly copied from https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial/blob/master/main_fine_tuning.py
# this file

def compute_measures(epoch: int, phase: str, dset_sizes, running_loss, c_matrix):
    if not os.path.exists("../Results/testresult.csv"):
        if not os.path.exists("../Results"):
            os.mkdir("../Results")
        log = open("../Results/testresult.csv", "a")
        lines = "epoch"+","+"phase"+","+"epoch_loss"+","+"tn"+","+"fp"+","+"fn"+","+"tp" + "\n"
        log.writelines(lines)
        log.close()
    tn, fp, fn, tp = c_matrix.ravel()
    epoch_loss = running_loss / dset_sizes[phase]
    epoch_acc = (tp + tn)/(tn+tp+fp+fn)


    a = np.transpose(pd.DataFrame(np.array([epoch, phase, epoch_loss, tn, fp, fn, tp])))
    a.columns = ["epoch", "phase", "epoch_loss", "tn", "fp", "fn", "tp"]
    new_obs = f"{epoch},{phase},{epoch_loss},{epoch_acc}\n"
    log = open("../Results/testresult.csv", "a")
    log.writelines(new_obs)
    log.close()
    return epoch_loss, epoch_acc


if __name__ == '__main__':

    # If you want to read more, transforms is a function from torchvision, and you can go read more here - http://pytorch.org/docs/master/torchvision/transforms.html
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = DATA_DIR
    dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                                   shuffle=True, num_workers=12)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes


    def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=50):
        since = time.time()
        best_loss_epoch = 1
        best_model = model
        best_loss = 100
        #asdf
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

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in dset_loaders[phase]:
                    inputs, labels = data
                    #those are needed, since I need the labels on cpu RAM, to compute the confusion matrix.
                    cpu_labels= copy.deepcopy(labels)
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
                    c_matrix= confusion_matrix(labels.data, preds)
                epoch_loss, epoch_acc = compute_measures(epoch, phase, dset_sizes, running_loss, c_matrix)
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

    model_ft = get_model("resnet", pretrained=False)
    #criterion = nn.CrossEntropyLoss()
    #loss adaptation for imbalanced learning
    weights = [1, 2]  # as class distribution. 1879 negativs, 110 positives.
    if torch.cuda.is_available():
        class_weights = torch.FloatTensor(weights).cuda()
    else:
        class_weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.0001)
    if torch.cuda.is_available():
        criterion.cuda()
        model_ft.cuda()

    # Run the functions and save the best model in the function model_ft.
    if os.path.exists("../Results/testresult.csv"):
        os.remove("../Results/testresult.csv")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=100)

    # Save model
    torch.save(model_ft.state_dict(), "../Models/fine_tuned_best_model.pt")
