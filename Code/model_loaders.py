from torchvision import models
import torch.nn as nn
from collections import OrderedDict
import torch

def get_model(model_name:str, pretrained=True, params_path=None):
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

class FilterModel(nn.Module):
    def __init__(self, model, threshold:int, num_classes):
        super(FilterModel, self).__init__()
        #the bikelane model is not getting trained, only the rim_layer
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.threshold = threshold
        self.rim_layer = nn.Linear(model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.model.fc(x)
        #convert model outputs into probabilities, transpose it such that the second row becomes the list of predicitons
        #for class 2 then take the second row
        y = y.softmax(1).t()[1]
        mask = y < self.threshold
        x = self.rim_layer(x)
        x[mask] = torch.tensor([[1, 0]], dtype=torch.float32, requires_grad=False)
        return x

    def get_trainable_params(self):
        self.rim_layer.parameters()

"""
model1 = get_model("resnet34", pretrained=True)
model2 = get_model("resnet50", pretrained=False)
model3 = get_model("resnet18", pretrained=False)


model1.state_dict()['conv1.weight'][0][0][0]
model2.state_dict()['conv1.weight'][0][0][0]
"""