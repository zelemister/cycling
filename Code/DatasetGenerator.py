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


class load_dataset(Dataset):
    def __init__(self, task: str, phase: str, set: str, transform: transforms, oversamplingrate: float, split: float,
                 resolution=256, model_name="resnet34", one_overoversampling=1):
        """
        :param task: string; either "bikelane", "rim", or "one_shot", "full_data"
        :param phase: string; either "train" or "test", this does not mean the training split, but the 9000/3000 split of the images, maybe
        :param set: either "train" or "val"
        :param transform: transformationfunction
        """
        self.set = set
        self.split = split
        self.rate = oversamplingrate
        self.transform = transform
        self.task = task
        self.model_name = model_name

        if "resnet" in model_name:
            self.resolution = resolution
        elif model_name == "transformer":
            self.resolution = 224
        if os.path.exists("../Images" + "_" + str(resolution) + "/" + phase):
            self.image_folder = "../Images" + "_" + str(resolution) + "/" + phase
        else: self.image_folder="../Images_256"
        labels = pd.read_csv("../labeling_clean.csv")
        labels_s = pd.read_csv("../labeling_steffen.csv")
        # labels_d = pd.read_csv("../labeling_daniel.csv")
        labels_rim = pd.read_csv("../RIM_Labels.csv")

        name_list = []
        label_list = []
        transformed_rim_names = ["x_" + str(x) + ".png" for x in labels_rim["NR_STR,C,254"]]
        for image in os.listdir(self.image_folder):
            label = -1
            if image in transformed_rim_names:
                label = 2
            elif image[:-4] in labels.Name.values:
                label = int(labels.loc[labels["Name"] == image[:-4], "Label"].values)
            elif image in labels_s["Name"].values:
                if not isnan(labels_s.loc[labels_s["Name"] == image, "Label"]):
                    label = int(labels_s.loc[labels_s["Name"] == image, "Label"])
            if label in [0, 1, 2]:
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

        if one_overoversampling>1 and isinstance(one_overoversampling, int) and set == "train":
            ones = self.dataset[self.dataset["Label"]==1]
            data_list = [self.dataset]
            while one_overoversampling > 1:
                data_list.append(ones)
                one_overoversampling -= 1
            self.dataset = pd.concat(data_list)

        self.dataset = self.dataset.reset_index(drop=True)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = self.dataset.loc[item].Name
        image = PIL.Image.open(os.path.join(self.image_folder, image))
        label = self.dataset.loc[item].Label

        if self.set == "train":
            if random.random() < 1 - (1 / self.rate):
                if self.transform:
                    image = self.transform(image)

        image = get_transformer("normalize", resolution=self.resolution)(image)

        return {"Image": image, "Label": label}

    def __len__(self):
        return self.dataset.__len__()


# this is testcode to show that batches generate new images for each time a dataloader loads a new batch
"""
random.seed(123456)
data = load_dataset(task="one_shot", phase="train", set="train", transform=get_transformer("rotations"),
                     oversamplingrate=2, split=0)
#data2 = load_dataset(task="rim", phase="train", set="train", transform=get_transformer("rotations"),
#                     oversamplingrate=2, split=0)

dset_loader = DataLoader(data, batch_size=10, shuffle=False)

model = get_model("transformer", pretrained=True)
for batch in dset_loader:
    out = model(batch["Image"])
    print(out)
    break
for i in range(5):
    dset_loader = DataLoader(data, batch_size=1, shuffle=False)
    for batch in dset_loader:
        if batch["Label"] == 1:
            img=transforms.ToPILImage()(batch["Image"].squeeze(0))
            img.show()
        #break
"""