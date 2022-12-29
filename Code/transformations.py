from torchvision import transforms
import torch
import random


def get_transformer(name: str, resolution = 256):
    if name == "rotations":
        transformer = transforms.Compose([transforms.ToTensor(),
                                          # generate a random number between 0 and 1, then multiply by 360 for a
                                          # random number between 0 and 360
                                          transforms.RandomRotation(degrees=random.random() * 360),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToPILImage()])
    elif name == "normalize":
        transformer = transforms.Compose([
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transformer
