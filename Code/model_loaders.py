from torchvision import models
import torch.nn as nn
from collections import OrderedDict

def get_model(model_name:str):
    #there are a couple ressources for finetuning a pretrained model
    #https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419
    #https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial
    num_classes = 2
    if model_name == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "transformer":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights)
        num_ftrs = model.heads.head.in_features
        model.heads = nn.Sequential(OrderedDict([('head', nn.Linear(num_ftrs, num_classes))]))
    return model
