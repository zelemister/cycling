import pandas as pd
import torch.cuda
from sklearn.model_selection import StratifiedKFold
from DatasetGenerator import load_dataset
from transformations import get_transformer
from model_loaders import get_model
from DatasetGenerator import get_model
from torch.utils.data import dataloader, SubsetRandomSampler
import numpy as np
from training import train_epoch, val_epoch
import os
import torch.optim as optim
import torch.nn as nn
import argparse


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
    optimizer = optimizer_keys[payload["optimizer"]]
    lr = payload["lr"]
    results_folder = payload["results_folder"]
    # get model
    model = get_model(model_name, pretrained=pretrained)

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

    # get weights
    weights = torch.FloatTensor(weights)
    weights.to(device)

    # get optimizer
    if params == "full":
        optimizer = optimizer(model.parameters(), lr=lr)
    else:
        if "resnet" in model_name:
            optimizer = optimizer(model.fc.parameters(), lr=lr)
        elif model_name == "transformer":
            optimizer = optimizer(model.heads.parameters(), lr=lr)
    optimizer.to(device)

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
    elif resolution == 512:
        batch_size = 16

    return model, data, device, optimizer, loss_fn, results_folder, batch_size, min_epochs, max_patience


def cross_validation(payload):
    model, data, device, optimizer, loss_fn, results_folder, batch_size, min_epochs, max_patience = parse_payload(
        payload)

    # create KFold Object
    splits = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold = 0
    history = {"Test_AUC": [], "Test_Loss": [], "Test_acc": [], "training_progress": []}
    for train_index, test_index in splits.split(X=data.dataset["Name"], y=data.dataset["Label"]):
        fold += 1
        train_subsampler = SubsetRandomSampler(train_index)
        test_subsampler = SubsetRandomSampler(test_index)
        train_loader = dataloader(data, batch_size=batch_size, shuffle=True, num_workers=12, sampler=train_subsampler)
        test_loader = dataloader(data, batch_size=batch_size, shuffle=True, num_workers=12, sampler=test_subsampler)
        epoch = 0
        training_progress = {"epoch": [], "train_loss": [], "train_auc": [], "train_acc": [], "val_loss": [],
                             "val_auc": [], "val_acc": []}
        best_auc = 0
        corresponding_loss = 100
        corresponding_acc = 0

        patience_counter = 0
        patience_max = 25
        while patience_counter <= patience_max or epoch <= min_epochs:
            epoch += 1
            train_loss, train_auc, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device=device)
            val_loss, val_auc, val_acc = val_epoch(model, test_loader, optimizer, loss_fn, device=device)
            if val_auc <= best_auc:
                patience_counter += 1
            else:
                best_auc = val_auc
                corresponding_loss = val_loss
                corresponding_acc = val_acc
                patience_counter = 0

            best_auc = max(val_auc, best_auc)
            training_progress["epoch"].append(epoch)
            training_progress["train_loss"].append(train_loss)
            training_progress["train_auc"].append(train_auc)
            training_progress["train_acc"].append(train_acc)
            training_progress["val_loss"].append(val_loss)
            training_progress["val_auc"].append(val_auc)
            training_progress["val_acc"].append(val_acc)
            print(f"Epoch: {epoch}, Val_Loss: {val_loss}, Val_AUC: {val_auc}, Patience: {patience_counter}")
        print("/n", "-" * 10)
        print(f"Fold: {fold}, AUC: {best_auc}, Loss: {corresponding_loss}, ACC: {corresponding_acc}")
        history["Test_AUC"].append(best_auc)
        history["Test_Loss"].append(corresponding_loss)
        history["Test_acc"].append(corresponding_acc)
        history["training_progress"].append(training_progress)

    # save the results
    for i in len(history["training_progress"]):
        temp = pd.DataFrame(history["training_progress"][i])
        temp.to_csv(os.path.join(results_folder, f"fold_{i + 1}.csv"))
    auc = np.mean(history["Test_AUC"])
    loss = np.mean(history["Test_Loss"])
    acc = np.mean(history["Test_acc"])
    return auc, loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="Default_Name")
    parser.add_argument('--min_epochs', type=int, default=80)
    parser.add_argument('--max_patience', type=int, default=25)
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

    payload = {"min_epochs": args.min_epochs, "max_patience": args.max_patience,
               "resolution": args.resolution, "transformation": args.transformation, "task": args.task,
               "model": args.model, "pretrained": args.pretrained, "params": args.params, "weights": args.weights,
               "optimizer": args.optimizer, "lr": args.lr, "results_folder": folder}
    auc, loss, acc = cross_validation(payload)
    del payload["results_folder"]
    payload["auc"] = auc
    payload["loss"] = loss
    payload["acc"] = acc

    print(auc, loss, acc)
    result = pd.DataFrame(payload, index=[0])
    result.to_csv(os.path.join(folder, "results.csv"))
