import os
import numpy
import torch
from PIL import Image
from torchvision import transforms
import pandas


def get_transformer(
        size: int = 256,
        rotate: bool = False
):
    transforms.Compose(
        [
        transforms.CenterCrop(size = (size, size)),
        transforms.ToTensor(),

        #@Daniel: as far as I have seen, all Image Data should be normalized such, that color values are between [0,1], std = 1
        #those normalization numbers are from Imagenet, which would only be ideal if we used a model pretrained on imagenet.
        #otherwise we should normalize using our datas means and stds.
        transforms.Normalize(
            [x / 255 for x in [125.3, 123.0, 113.9]],
            [x / 255 for x in [63.0, 62.1, 66.7]]
        )
        ]
    )


def get_dataloaders(
        batch_size_train: int = 128,
        batch_size_test: int = 128,
        batch_size_predict: int = 128,
        n_workers: int = 4
        ):
    transformer = get_transformer(size = 256)


    #sets that have to be defined and returned for the final experiment.


    #The working directory should be the "code" folder, but the location of this file and the working directory might change in the future
    labels = pandas.read_csv("../labeling_clean.csv")

    trainset_roads
    testset_roads
    trainset_RIM
    testset_RIM
    labelset_final
