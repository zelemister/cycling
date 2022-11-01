import pandas
import os
import shutil
import random
#this dataset sorter seems to be necessary, since apparently

folder = "../Data"
if not os.path.exists("../Data/bikelane"):
    os.mkdir("../Data/bikelane")
if not os.path.exists("../Data/no_bikelane"):
    os.mkdir("../Data/no_bikelane")

labels = pandas.read_csv("../labeling_clean.csv")
for i in range(labels.__len__()):

    old = folder + "/" + labels.loc[i, "Name"] + ".png"
    if os.path.exists(old):
        if labels.loc[i, "Label"] == 0:
            new = folder + "/no_bikelane/" + labels.loc[i, "Name"] + ".png"
        elif labels.loc[i, "Label"] == 1:
            new = folder + "/bikelane/" + labels.loc[i, "Name"] + ".png"

        if new != old:
            shutil.move(old, new)

