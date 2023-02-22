import copy

import pandas as pd
import torch.cuda
from sklearn.model_selection import StratifiedKFold
from DatasetGenerator import load_dataset
from transformations import get_transformer
from model_loaders import get_model, FilterModel
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from training import train_epoch, val_epoch
import os
import torch.optim as optim
import torch.nn as nn
import argparse
import random
from pathlib import Path

class Model_Optim_Gen:
    def __init__(self, device, optimizer_fn, model_name="resnet34", pretrained=True, params="full", lr=0.001, stages=1, quantile=0.9):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.optimizer_fn = optimizer_fn
        self.params = params
        self.lr = lr
        self.stages=stages
        self.num_classes = 2
        self.k = 1
        self.quantile = quantile
        self.path = Path("../Results/Bikelane_tunedFor2Phase/Config_1/")
    def new_model(self):
        if self.stages ==1:
            model = get_model(self.model_name, pretrained=self.pretrained)
            model.to(self.device)
        elif self.stages==2:
            temp_model = get_model(self.model_name)
            temp_model.load_state_dict(torch.load(self.path.joinpath(f"model_{self.k}.pt"),
                                             map_location=self.device))
            predictions = pd.read_csv(self.path.joinpath(f"predictions_{self.k}.csv"))
            predictions = predictions.sort_values(by='prediction', ascending=False)
            indices = [i for i in range(len(predictions)) if predictions.label[i] == 1]
            threshold_index = np.quantile(indices, self.quantile)
            threshold = predictions.prediction[round(threshold_index)]
            model = FilterModel(temp_model, threshold=threshold, num_classes=self.num_classes)
            self.k +=1
        return model

    def new_optim(self, model):
        if self.stages==1:
            if self.params == "full":
                optimizer = self.optimizer_fn(model.parameters(), lr=self.lr)
            else:
                if "resnet" in self.model_name:
                    optimizer = self.optimizer_fn(model.fc.parameters(), lr=self.lr)
                elif self.model_name == "transformer":
                    optimizer = self.optimizer_fn(model.heads.parameters(), lr=self.lr)
        elif self.stages==2:
            optimizer = self.optimizer_fn(model.rim_layer.parameters(), lr=self.lr)
        return optimizer


def parse_payload(payload):
    optimizer_keys = {"RMSProp": optim.RMSprop,
                      "SGD": optim.SGD,
                      "Adam": optim.Adam}
    min_epochs = payload["min_epochs"]
    max_patience = payload["max_patience"]
    oversampling_rate = payload["oversampling_rate"]
    resolution = payload["resolution"]
    transformation = get_transformer(payload["transformation"], resolution)
    task = payload["task"]
    model_name = payload["model"]
    pretrained = payload["pretrained"]
    params = payload["params"]
    weights = [1, payload["weights"]]
    optimizer_fn = optimizer_keys[payload["optimizer"]]
    lr = payload["lr"]
    results_folder = payload["results_folder"]
    stages = payload["stages"]
    quantile = payload["quantile"]
    path = Path(results_folder)
    if not path.exists():
        path.mkdir(parents=True)
    i=1
    while path.joinpath(f"Config_{i}").exists():
        i+=1
    results_folder = path.joinpath(f"Config_{i}")
    results_folder.mkdir(parents=True)


    # get dataset
    data_payload = {"task": task, "phase": "train", "set": "train", "transform": transformation,
                    "oversamplingrate": oversampling_rate, "split": 0, "resolution": resolution,
                    "model_name": model_name}
    data = load_dataset(**data_payload)
    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    generator = Model_Optim_Gen(device, optimizer_fn, model_name=model_name, pretrained=pretrained, params=params,
                                lr=lr, stages=stages, quantile=quantile)

    # get weights
    weights = torch.FloatTensor(weights)
    weights.to(device)

    # get loss function
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="mean")
    loss_fn.to(device)

    # get batch_size
    if resolution == 256:
        if model_name in ["resnet34", "resnet18"]:
            batch_size = 64
        elif model_name == "resnet50":
            batch_size = 32
        elif model_name == "transformer":
            batch_size = 16
        else:
            batch_size = 64
    elif resolution == 512:
        batch_size = 16

    return generator, data, device, loss_fn, results_folder, batch_size, min_epochs, max_patience

def save_results(history, folder, logging=True):
    if logging:
        for i in range(len(history["training_progress"])):
            temp = pd.DataFrame(history["training_progress"][i])
            temp.to_csv(os.path.join(folder, f"fold_{i + 1}.csv"))

def get_prediction_list(model, trainloader, testloader, device, folder, fold):
    pred_list=[]
    label_list=[]
    name_list=[]
    for data in trainloader:
        images = data["Image"]
        labels = data["Label"]
        names = data["Name"]

        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = model(images)
        #predicted score for class 1
        predictions = output.softmax(1).t()[1]
        pred_list += predictions.tolist()
        label_list +=labels.tolist()
        name_list +=names

    train_preds = pd.DataFrame({"label": label_list, "prediction":pred_list, "name":name_list})
    train_preds.to_csv(folder.joinpath(f"predictions_{fold}.csv"))
    pred_list=[]
    label_list=[]
    name_list=[]

    for data in testloader:
        images = data["Image"]
        labels = data["Label"]
        names = data["Name"]

        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = model(images)
        #predicted score for class 1
        predictions = output.softmax(1).t()[1]
        pred_list += predictions.tolist()
        label_list +=labels.tolist()
        name_list +=names

    val_preds = pd.DataFrame({"label": label_list, "prediction":pred_list, "name":name_list})
    val_preds.to_csv(folder.joinpath(f"val_predictions_{fold}.csv"))

def cross_validation(payload, seed=0, k=5):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    generator, data, device, loss_fn, results_folder, batch_size, min_epochs, max_patience = parse_payload(
        payload)
    train_data = copy.deepcopy(data)
    test_data = copy.deepcopy(data)
    test_data.set = "val"
    # create KFold Object
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold = 0
    history = {"Test_AUC": [], "Test_Loss": [], "Test_acc": [], "training_progress": []}
    for train_index, test_index in splits.split(X=data.dataset["Name"], y=data.dataset["Label"]):
        fold += 1
        print("-"*20, f"FOLD {fold}", "-"*20)
        train_subsampler = SubsetRandomSampler(train_index)
        test_subsampler = SubsetRandomSampler(test_index)
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=12, sampler=train_subsampler)
        test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=12, sampler=test_subsampler)
        model = generator.new_model()
        optimizer = generator.new_optim(model)
        training_progress = {"epoch": [], "train_loss": [], "train_auc": [], "train_acc": [], "val_loss": [],
                             "val_auc": [], "val_acc": []}

        epoch = 0
        best_loss = 1000
        corresponding_auc = 0
        corresponding_acc = 0

        patience_counter = 0
        while (patience_counter < max_patience or epoch <= min_epochs) and epoch <= 300:
            epoch += 1
            train_loss, train_auc, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device=device)
            val_loss, val_auc, val_acc = val_epoch(model, test_loader, loss_fn, device=device)
            if val_loss >= best_loss:
                patience_counter += 1
            else:
                best_loss = val_loss
                corresponding_auc = val_auc
                corresponding_acc = val_acc
                patience_counter = 0
                if k==1 or payload["save_model"]:
                    best_model=copy.deepcopy(model)

            training_progress["epoch"].append(epoch)
            training_progress["train_loss"].append(train_loss)
            training_progress["train_auc"].append(train_auc)
            training_progress["train_acc"].append(train_acc)
            training_progress["val_loss"].append(val_loss)
            training_progress["val_auc"].append(val_auc)
            training_progress["val_acc"].append(val_acc)
            print(f"Epoch: {epoch}, Val_Loss: {round(val_loss,3)}, Val_AUC: {round(val_auc,3)}, Patience: {patience_counter}")

        print("-" * 10)
        print(f"Fold: {fold}, AUC: {round(corresponding_auc, 3)}, Loss: {round(best_loss,3)}, ACC: {round(corresponding_acc,3)}")
        history["Test_AUC"].append(corresponding_auc)
        history["Test_Loss"].append(best_loss)
        history["Test_acc"].append(corresponding_acc)
        history["training_progress"].append(training_progress)
        if payload["save_model"]:
            torch.save(best_model.state_dict(), os.path.join(results_folder, f"model_{fold}.pt"))
            predictions = get_prediction_list(best_model, train_loader, test_loader,
                                              device=device, folder=results_folder, fold=fold)

        if k==1:
            save_results(history, results_folder, payload["logging"])
            auc = np.mean(history["Test_AUC"])
            loss = np.mean(history["Test_Loss"])
            acc = np.mean(history["Test_acc"])
            torch.save(best_model.state_dict(), os.path.join(results_folder, "trained_model.pt"))
            return auc, loss, acc
    # save the results
    save_results(history, results_folder,payload["logging"])
    auc = np.mean(history["Test_AUC"])
    loss = np.mean(history["Test_Loss"])
    acc = np.mean(history["Test_acc"])
    return auc, loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="Default_Name")
    parser.add_argument('--min_epochs', type=int, default=70)
    parser.add_argument('--max_patience', type=int, default=30)
    parser.add_argument('--oversampling_rate', type=int, default=10)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--transformation', choices=["rotations", "colorJitter", "gBlur", "all"], default="rotations")
    parser.add_argument('--task', type=str, default="bikelane")
    parser.add_argument('--model', type=str, default="resnet34")
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--params', choices=["full", "head"], default="full")
    parser.add_argument('--weights', type=int, default=1)
    parser.add_argument('--optimizer', choices=["RMSProp", "SGD", "Adam"], default="RMSProp")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--stages', type=int, default=1)
    parser.add_argument('--k', type=int, default=5) #if k==1, then the model is trained instead
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--quantile', type=float, default=0.9)

    args = parser.parse_args()

    # get the folder, where to save results to
    folder = "../Results/" + args.name
    if not os.path.exists(os.path.split(folder)[0]):
        os.mkdir(os.path.split(folder)[0])
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        folder_changed = folder
        i = 2
        while os.path.exists(folder_changed):
            folder_changed = folder + "_" + str(i)
            i += 1
        folder = folder_changed
        os.mkdir(folder)

    # payload = {"min_epochs": args.min_epochs, "max_patience": args.max_patience, "oversampling_rate":args.oversampling_rate,
    #            "resolution": args.resolution, "transformation": args.transformation, "task": args.task,
    #            "model": args.model, "pretrained": args.pretrained, "params": args.params, "weights": args.weights,
    #            "optimizer": args.optimizer, "lr": args.lr, "stages": args.stages, "results_folder": folder}
    payload = {"min_epochs": args.min_epochs, "max_patience": args.max_patience, "oversampling_rate":args.oversampling_rate,
               "resolution": args.resolution, "transformation": args.transformation, "task": args.task,
               "model": args.model, "pretrained": args.pretrained, "params": args.params, "weights": args.weights,
               "optimizer": args.optimizer, "lr": args.lr, "stages": 1, "results_folder": folder, "logging":True,
               "save_model":args.save_model, "quantile":args.quantile}

    auc, loss, acc = cross_validation(payload, k=args.k)

    payload["auc"] = auc
    payload["loss"] = loss
    payload["acc"] = acc

    print(auc, loss, acc)
    result = pd.DataFrame(payload, index=[0])
    result.to_csv(os.path.join(folder, "results.csv"))
    #torch.save(model.state_dict(), os.path.join(folder, "trained_model.pt"))