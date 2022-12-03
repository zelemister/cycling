import pandas
import os
import shutil
import random
from math import isnan
#this dataset sorter seems to be necessary, since apparently
random.seed(123456)
images = "../Images"
folder = "../Data"
if not os.path.exists(folder):
    os.mkdir(folder)
bike_folder = folder + "/bikelane"
rim_folder = folder + "/rim"

for name in [bike_folder, rim_folder]:
    if not os.path.exists(name):
        os.mkdir(name)
    for set in ["/train", "/val"]:
        if not os.path.exists(name + set):
            os.mkdir(name+set)
        for label in ["/0","/1"]:
            if not os.path.exists(name+set+label):
                os.mkdir(name+set+label)

labels = pandas.read_csv("../labeling_clean.csv")
#labels_d = pandas.read_csv("../labeling_daniel.csv")
labels_s = pandas.read_csv("../labeling_steffen.csv")

"""
#sort Images into the dataset
for image in os.listdir(images):
    label = -1
    if image[:-4] in labels.Name.values:
        label = int(labels.loc[labels["Name"] == image[:-4], "Label"].values)
    #elif image in labels_d["Name"].values:
        #if not isnan(labels_d.loc[labels_d["Name"]==image, "Label"]):
        #    if int(labels_d.loc[labels_d["Name"]==image, "Label"]) in [0,1]:

        #    label = int(labels_d.loc[labels_d["Name"]==image, "Label"])
    elif image in labels_s["Name"].values:
        if not isnan(labels_s.loc[labels_s["Name"]==image, "Label"]):
            if int(labels_s.loc[labels_s["Name"]==image, "Label"]) in [0,1]:
                label = int(labels_s.loc[labels_s["Name"] == image, "Label"])
    if label in [0,1]:
        if random.random()<0.2:
            shutil.copyfile(images+"/"+image, bike_folder + "/val/" + str(label) + "/" + image)
        else:
            shutil.copyfile(images+"/"+image, bike_folder + "/train/" + str(label)+ "/" + image)
"""


rims = pandas.read_csv(os.path.join("..", "RIM_Labels.csv"))
for intersection in rims["NR_STR,C,254"]:
    file_name = "x_" + str(intersection) + ".png"
    if random.random() < 0.2:
        shutil.copyfile(os.path.join(images, file_name), os.path.join(rim_folder, "val", "1", file_name))
    else:
        shutil.copyfile(os.path.join(images, file_name), os.path.join(rim_folder, "train", "1", file_name))


"""for i in range(labels.__len__()):
    old = folder + "/" + labels.loc[i, "Name"] + ".png"

    if os.path.exists(old):
        if labels.loc[i, "Label"] == 0:
            new = "0/" + labels.loc[i, "Name"] + ".png"
        elif labels.loc[i, "Label"] == 1:
            new = "1/" + labels.loc[i, "Name"] + ".png"

        if random.random() < 0.2:
            shutil.copyfile(old, bike_folder + "/val/" + new)
            print(f"Copied File {new[2:]}")
        else:
            shutil.copyfile(old, bike_folder + "/train/" + new)
            print(f"Copied File {new[2:]}")
"""