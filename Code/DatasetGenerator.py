from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
from math import isnan
import random
from transformations import get_transformer
import torch
import PIL.Image

from torch.utils.data import DataLoader


class load_dataset(Dataset):
    def __init__(self, task: str, phase: str, set: str, transform: transforms, oversamplingrate: float, split: float):
        """
        :param task: string; either "bikelane" or "rim"
        :param phase: string; either "train" or "test", this does not mean the training split, but the 9000/3000 split of the images, maybe
        :param set: either "train" or "val"
        :param transform: transformationfunction
        """
        self.set = set
        self.split = split
        self.rate = oversamplingrate
        self.transform = transform
        self.task = task
        if os.path.exists("../" + phase):
            self.image_folder = "../" + phase
        else:
            self.image_folder = "../Images"

        labels = pd.read_csv("../labeling_clean.csv")
        labels_s = pd.read_csv("../labeling_steffen.csv")
        # labels_d = pd.read_csv("../labeling_daniel.csv")
        labels_rim = pd.read_csv("../RIM_Labels.csv")

        name_list = []
        label_list = []
        for image in os.listdir(self.image_folder):
            label = -1
            if image[:-4] in labels.Name.values:
                label = int(labels.loc[labels["Name"] == image[:-4], "Label"].values)
            # RIMs are also bikelanes
            elif image[:-4] in labels_rim["NR_STR,C,254"]:
                label = 2
            # elif image in labels_d["Name"].values:
            # if not isnan(labels_d.loc[labels_d["Name"]==image, "Label"]):
            #    if int(labels_d.loc[labels_d["Name"]==image, "Label"]) in [0,1]:

            #    label = int(labels_d.loc[labels_d["Name"]==image, "Label"])
            #    if label == 2: label = 2 #rims are also bikelanes
            elif image in labels_s["Name"].values:
                if not isnan(labels_s.loc[labels_s["Name"] == image, "Label"]):
                    label = int(labels_s.loc[labels_s["Name"] == image, "Label"])
                    if label == 2: label = 2  # rims are also bikelanes
            # sort randomly into val set and train set
            if label in [0, 1, 2]:
                if random.random() < self.split:
                    if self.set == "val":
                        name_list.append(image)
                        label_list.append(label)
                else:
                    if self.set == "train":
                        name_list.append(image)
                        label_list.append(label)

        if self.task == "bikelane":
            # this should set all "2" to "1", such that rims count as bikelanes
            label_list = [min(1, label_list[i]) for i in range(len(label_list))]
            self.dataset = pd.DataFrame({"Name": name_list, "Label": label_list})
        elif self.task == "rim":
            """
            We select all indizes that are equal to 1 or 2 (bikelanes or rims) which what we are interested in in task 2
            Then we get the sublist which only contains the labels of those images
            Then we subtract 1 from those labels, such that our task is differing 0 and 1
            Then we get the sublist which only contains the names of those images
            Then we create our self.Dataset using those created lists.
            """
            indizes = [i for i, j in enumerate(label_list) if j in [1,2]]
            rim_labels = [label_list[i] for i in indizes]
            correct_labels = [rim_labels[i]-1 for i in range(len(rim_labels))]
            rim_names = [name_list[i] for i in indizes]
            self.dataset = pd.DataFrame({"Name": rim_names, "Label": correct_labels})
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = self.dataset.loc[item].Name
        image = PIL.Image.open(os.path.join(self.image_folder, image))
        label = self.dataset.loc[item].Label

        if self.set == "train":
            if random.random() < 1 - (1 / self.rate):
                image = get_transformer("rotations")(image)

        if self.transform:
            image = self.transform(image)
        return {"Image": image, "Label": label}

    def __len__(self):
        return self.dataset.__len__()


# this is testcode to show that batches generate new images for each time a dataloader loads a new batch
random.seed(123456)
data = load_dataset(task="rim", phase="train", set="train", transform=transforms.ToTensor(),
                    oversamplingrate=2, split=0)
dset_loader = DataLoader(data, batch_size=1, shuffle=False)

for i in range(5):
    dset_loader = DataLoader(data, batch_size=1, shuffle=False)
    for batch in dset_loader:
        img=transforms.ToPILImage()(batch["Image"].squeeze(0))
        img.show()
        break
