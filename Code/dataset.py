import os

import PIL.Image
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# implementation of custom dataset code from
# https://github.com/rachellea/pytorch-computer-vision/blob/master/load_dataset/custom_tiny.py article from
# https://towardsdatascience.com/building-custom-image-datasets-in-pytorch-15ba855b47cb


class CyclingData(Dataset):
    def __init__(self, task):
        """Cycling Dataset for the imagedata that we are using.
        'task' can be any one of:  'red' to train the bikelanes model
                                   'rim' to train the rim model
                                   'pred' classify the unlabeled obs """
        self.task = task

        self.datafolder = "../Data"
        red_labels_file = "../labeling_clean.csv"
        red_labels = pd.read_csv(red_labels_file)
        # missing: rim labels
        # missing implied unlabeled indices

        df = pd.DataFrame()
        if task == 'red':

            # for each image that was labeled by hand, check if it's in the Data (some of those images were withheld
            # so are not included) then add it to the images of the dataset
            for i in range(red_labels.__len__()):
                image_path = self.datafolder + '/' + red_labels.loc[i, "Name"] + '.png'
                if os.path.exists(image_path):
                    df = df.append({'Name': red_labels.loc[i, "Name"], 'Label': red_labels.loc[i, "label"]},
                                   ignore_index=True)
        if task == 'rim':
            "unfilled"
        if task == 'pred':
            "unfilled"

        self.df = df

    def __len__(self):
        """return total number of images in this Dataset"""
        return len(self.df)

    def __getitem__(self, idx):
        """return the image as given by index"""
        imagefile = self.datafolder + '/' + self.df.loc[idx, "Name"] + '.png'
        image = PIL.Image.open(imagefile).convert('RGB')
        label = torch.Tensor(self.df.loc[idx, "Label"])

        sample = {'data': image,
                  'label': label,
                  'img_idx': idx}
        return sample
