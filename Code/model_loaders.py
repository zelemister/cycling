from torchvision import models
import torch.nn as nn
from collections import OrderedDict

def get_model(model_name:str, pretrained=True):
    #there are a couple ressources for finetuning a pretrained model
    #https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419
    #https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial
    num_classes = 2
    if "resnet" in model_name:
        if model_name == "resnet34":
            if pretrained:
                model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                model = models.resnet34()
        elif model_name == "resnet50":
            if pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                model = models.resnet50()
        elif model_name == "resnet18":
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                model = models.resnet18()

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "transformer":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        num_ftrs = model.heads.head.in_features
        model.heads = nn.Sequential(OrderedDict([('head', nn.Linear(num_ftrs, num_classes))]))
    return model
model1 = get_model("resnet34", pretrained=True)
model2 = get_model("resnet50", pretrained=False)
model3 = get_model("resnet18", pretrained=False)


model1.state_dict()['conv1.weight'][0][0][0]
model2.state_dict()['conv1.weight'][0][0][0]