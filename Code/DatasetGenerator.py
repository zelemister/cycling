from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
from math import isnan
import random
from transformations import get_transformer
import torch
import PIL.Image
from model_loaders import get_model
from torch.utils.data import DataLoader
import numpy as np

class load_dataset(Dataset):
    def __init__(self, task: str, phase: str, set: str, transform: transforms, oversamplingrate: float, split: float,
                 resolution=256, model_name="resnet34"):
        """
        :param task: string; either "bikelane", "rim", or "one_shot", "full_data" with the following explanations:
                            bikelane: all bikelanes and rims are set to 1, we want to identify cycling infrastructure
                            rim: never used, but it set all rims to 1 and all bikelanes to 0 leading to a smaller dataset
                            one_shot: sets all rim to 1 all non rims to 0, working with the full dataset
                            full_data: never used, keeps all labels as they are, leading to 3 total classes
        :param phase: string; either "train" or "test", this does not mean the training split, but the 9000/3000 split
                            of the images, this was relevant for our work, for subsequent projects it's not important
        :param set: string; either 'train' or 'val', depending on the 'split' it uses only a subset of the data, and
                            depending on the
        :param transform: transformationfunction
        """
        self.set = set
        self.split = split
        self.rate = oversamplingrate
        self.transform = transform
        self.task = task
        self.model_name = model_name

        if model_name == "transformer":
            self.resolution = 224
        else:
            self.resolution=resolution
        if os.path.exists("../Images" + "_" + str(resolution) + "/" + phase):
            self.image_folder = "../Images" + "_" + str(resolution) + "/" + phase
        else: self.image_folder="../Images_" + str(resolution)

        data = pd.read_csv("../labels_complete.csv")
        if phase in ["train", "test"]:
            data = data.loc[(data.phase == phase) & [not x for x in np.isnan(data["Label"])]]
            data.Label = data.Label.astype(int)
            label_list = data.Label.to_list()
            name_list = data.Name.to_list()
        elif phase == "complete_data":
            data = data.loc[[not x for x in np.isnan(data["Label"])]]
            data.Label = data.Label.astype(int)
            label_list = data.Label.to_list()
            name_list = data.Name.to_list()
        if resolution ==512:
            corrupted_file = name_list.index("x_9010063.png")
            del name_list[corrupted_file]
            del label_list[corrupted_file]
        if self.task == "bikelane":
            # this should set all "2" to "1", such that rims count as bikelanes
            label_list = [min(1, label_list[i]) for i in range(len(label_list))]
            self.dataset = pd.DataFrame({"Name": name_list, "Label": label_list})

        elif self.task == "rim":
            """
            This setting is not used, as it was deemed too complicated
            We select all indizes that are equal to 1 or 2 (bikelanes or rims) which what we are interested in in task 2
            Then we get the sublist which only contains the labels of those images
            Then we subtract 1 from those labels, such that our task is differing 0 and 1
            Then we get the sublist which only contains the names of those images
            Then we create our self.Dataset using those created lists.
            """

            indizes = [i for i, j in enumerate(label_list) if j in [1, 2]]
            rim_labels = [label_list[i] for i in indizes]
            correct_labels = [rim_labels[i] - 1 for i in range(len(rim_labels))]
            rim_names = [name_list[i] for i in indizes]
            self.dataset = pd.DataFrame({"Name": rim_names, "Label": correct_labels})

        elif self.task == "one_shot":
            label_list = [0 if x == 1 else x for x in label_list]
            label_list = [1 if x == 2 else x for x in label_list]
            self.dataset = pd.DataFrame({"Name": name_list, "Label": label_list})

        elif self.task == "full_data":
            self.dataset = pd.DataFrame({"Name": name_list, "Label": label_list})

        if set == "train":
            self.dataset = self.dataset.sample(frac=max(split, 1 - split), random_state=42)
        elif set == "val":
            sample = self.dataset.sample(frac=max(split, 1 - split), random_state=42)
            self.dataset = self.dataset[~self.dataset.index.isin(sample.index)]

        self.dataset = self.dataset.reset_index(drop=True)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = self.dataset.loc[item].Name
        image = PIL.Image.open(os.path.join(self.image_folder, image))
        label = self.dataset.loc[item].Label

        if self.set == "train":
            if random.random() < self.rate:
                if self.transform:
                    image = self.transform(image)

        image = get_transformer("normalize", resolution=self.resolution)(image)

        return {"Image": image, "Label": label, "Name": self.dataset.loc[item].Name}

    def __len__(self):
        return self.dataset.__len__()


# this is testcode to show that batches generate new images for each time a dataloader loads a new batch
if __name__=="__main__":
    random.seed(123456)
    data = load_dataset(task="one_shot", phase="train", set="train", transform=get_transformer("rotations"),
                         oversamplingrate=0.5, split=0)
    #data2 = load_dataset(task="rim", phase="train", set="train", transform=get_transformer("rotations"),
    #                     oversamplingrate=2, split=0)
    data2 = load_dataset(task="one_shot", phase="complete_data", set="train", transform=get_transformer("rotations"),
                         oversamplingrate=0.5, split=0)

    dset_loader = DataLoader(data, batch_size=10, shuffle=False)

    model = get_model("resnet18", pretrained=True)
    for batch in dset_loader:
        out = model(batch["Image"])
        print(out)
    for i in range(5):
        dset_loader = DataLoader(data, batch_size=1, shuffle=False)
        for batch in dset_loader:
            if batch["Label"] == 1:
                img=transforms.ToPILImage()(batch["Image"].squeeze(0))
                img.show()
            #break