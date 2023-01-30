import copy

import pandas as pd
from torch.utils.data import DataLoader
from Crossvalidation import parse_payload
from training import val_epoch, train_epoch
import argparse
import os
import torch
from overlooked_images import generate_false_negative_list
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="Default_Name")
    parser.add_argument('--min_epochs', type=int, default=70)
    parser.add_argument('--max_patience', type=int, default=50)
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
               "optimizer": args.optimizer, "lr": args.lr, "stages": 1, "results_folder": folder, "logging":True}
    frac = 0.2
    generator, data, device, loss_fn, results_folder, batch_size, min_epochs, max_patience = parse_payload(payload)
    model = generator.new_model()
    optimizer = generator.new_optim(model)
    train_data = copy.deepcopy(data)
    train_data.dataset = train_data.dataset.sample(frac=max(frac, 1 - frac), random_state=42)
    test_data = copy.deepcopy(data)
    test_data.set= "val"
    test_sample = data.dataset.sample(frac=max(frac, 1 - frac), random_state=42)
    test_data.dataset[~test_data.dataset.index.isin(test_sample.index)]

    epoch=0
    patience_counter = 0
    best_loss = 10000
    corresponding_auc=0
    corresponding_acc=0
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=12)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=12)
    training_progress = {"epoch": [], "train_loss": [], "train_auc": [], "train_acc": [], "val_loss": [],
                         "val_auc": [], "val_acc": []}
    while patience_counter < max_patience or epoch <= min_epochs or epoch >= 300:
        epoch += 1
        train_loss, train_auc, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device=device)
        val_loss, val_auc, val_acc = val_epoch(model, test_loader, loss_fn, device=device)
        if val_loss >= best_loss:
            patience_counter += 1
        else:
            best_loss = val_loss
            corresponding_auc = val_auc
            corresponding_acc = val_acc
            best_model = copy.deepcopy(model)
            patience_counter = 0
        print(f"Epoch: {epoch}, Val_Loss: {round(val_loss,3)}, Val_AUC: {round(val_auc,3)}, Patience: {patience_counter}")

        training_progress["epoch"].append(epoch)
        training_progress["train_loss"].append(train_loss)
        training_progress["train_auc"].append(train_auc)
        training_progress["train_acc"].append(train_acc)
        training_progress["val_loss"].append(val_loss)
        training_progress["val_auc"].append(val_auc)
        training_progress["val_acc"].append(val_acc)

    temp = pd.DataFrame(training_progress)
    temp.to_csv(os.path.join(results_folder, "training_log.csv"))
    temp = pd.DataFrame({"Best_loss":best_loss, "Corresponding_AUC": corresponding_auc, "Corresponging_ACC":corresponding_acc}, index=[0])
    temp.to_csv(os.path.join(results_folder, "results.csv"))
    torch.save(best_model.state_dict(), os.path.join(results_folder, "trained_model.pt"))
    generate_false_negative_list(data=test_data, destination=results_folder, model=best_model)







