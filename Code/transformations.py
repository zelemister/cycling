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
    elif name == "colorJitter":
        transformer = transforms.Compose([transforms.ToTensor(),
                                          #could use random.random instead of fixed
                                          #could add option hue with max interval [-0.5, 0.5]
                                          transforms.ColorJitter(0.5, contrast=0.5, saturation=0.5)(),
                                          transforms.ToPILImage()])
    elif name == "gBlur":
        transformer = transforms.Compose([transforms.ToTensor(),
                                          # kernel size positive uneven number
                                          transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 200))(),
                                          transforms.ToPILImage()])
    elif name == "all":
        transformer = transforms.Compose([transforms.ToTensor(),
                                          # generate a random number between 0 and 1, then multiply by 360 for a
                                          # random number between 0 and 360
                                          transforms.RandomRotation(degrees=random.random() * 360),
                                          #default probability is p=0.5
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                                 saturation=0.5)(),
                                          transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 200))(),
                                          transforms.ToPILImage()])
    elif name == "grayscale":
        transformer = transforms.Compose([transforms.ToTensor(),
                                          #can be used to test impact of colour in image
                                          transforms.Grayscale(num_output_channels=3)(),
                                          transforms.ToPILImage()])
    elif name == "normalize":
        transformer = transforms.Compose([
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transformer
