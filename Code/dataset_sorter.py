import pandas
import os
import shutil
import random
#this dataset sorter seems to be necessary, since apparently

folder = "../Data"
if not os.path.exists(folder):
    os.mkdir(folder)

bike_folder = folder + "bikelane"
if not os.path.exists(folder + "bikelane"):
    os.mkdir(folder + "bikelane")

if not os.path.exists(bike_folder + "/train"):
    os.mkdir(bike_folder + "/train")

if not os.path.exists(bike_folder + "/val"):
    os.mkdir(bike_folder + "/val")

for name in [bike_folder + "/train", bike_folder + "/val"]:
    if not os.path.exists(name + "0"):
        os.mkdir(name + "0")
    if not os.path.exists(name + "1"):
        os.mkdir(name + "1")

labels = pandas.read_csv("../labeling_clean.csv")
for i in range(labels.__len__()):
    old = folder + "/" + labels.loc[i, "Name"] + ".png"

    if os.path.exists(old):
        if labels.loc[i, "Label"] == 0:
            new = "0/" + labels.loc[i, "Name"] + ".png"
        elif labels.loc[i, "Label"] == 1:
            new = "1/" + labels.loc[i, "Name"] + ".png"

        if random.random() < 0.2:
            shutil.copyfile(old, bike_folder + "/val/" + new)
        else:
            shutil.copyfile(old, bike_folder + "/train/" + new)


