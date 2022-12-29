from config_file import *
from model_loaders import get_model
from torchvision import datasets, transforms
from transformations import get_transformer
import os
from PIL import Image
import numpy.random as random
import torch
import pandas as pd
from DatasetGenerator import load_dataset
from torch.utils.data import Dataset


"""
The point of this file is to identify all the images that are falsely classified as 0
even though they are of class 1. Since we are mainly interested in identifying the 1 classifcated images,
and there aren't a lot of them, this seems to be feasible. 
To do this, we read the validation folder of the desired Task (bikelanes, rim),
run them through our model, and then use softmax to get predictions of labels.
Then we take those labels where the prediction is 0, even though the label is 1, and look at them individually.
This might also just take the "worst" predictions, aka, those where the model is the most certain that the picture is 0
even though it's one.
"""

def generate_false_negative_list(data:Dataset, destination:str, model):

    #read model
    model.cpu()
    model.eval()


    list=pd.DataFrame({"Name": [], "prob": []})

    #read in image, normalize it and run it through model
    for i in range(data.__len__()):
        name=data.dataset["Name"][i]
        img, label= data.__getitem__(i)["Image"], data.__getitem__(i)["Label"]
        if label==1:
            #model only take batches, and apparently unsqueeze does create a batch of length 1
            img= img.unsqueeze(0)
            with torch.no_grad():
                out = model(img)

            #this should convert modeloutput into probabilities, strip one layer of lists that accumulate,
            #and then take the second element, which should be the prob for label 1
            one_pred=torch.softmax(out,1)[0][1]

            #this should take the higher value, and return the index of that value, if the
            #index = 1, this means that it's the second class(1) if it's 0, that means it's the
            #first class(0)
            pred_label=torch.max(out, 1)[1]
            if pred_label==0:
                list.loc[len(list)]= [name, one_pred.item()]
    print()
    list.to_csv(destination + "/overlooked_images.csv")
"""
data_transforms = get_transformer("normalize")
payload = {"task": "bikelane", "phase": "train", "transform": data_transforms, "oversamplingrate": 10, "split": 0.2}
destination = "../Results/New_Oversampling_No_Weights_No_Pretrained_2"
model = get_model("resnet", pretrained=False)
data = load_dataset(**payload, set="val")
model.load_state_dict(torch.load(destination + "/fine_tuned_best_model.pt", map_location=torch.device("cpu")))
generate_false_negative_list(data, destination, model)
"""