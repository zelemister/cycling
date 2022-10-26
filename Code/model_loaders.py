from torchvision import models
import torch.nn as nn


def get_model():
    #there are a couple ressources for finetuning a pretrained model
    #https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419
    #https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial
    model = models.resnet50(weights= models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    num_classes = 2

    model.fc = nn.Linear(num_ftrs, 2)
    return model
