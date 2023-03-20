import pandas as pd
import torch.utils.data

from training import val_epoch
from DatasetGenerator import load_dataset
from model_loaders import get_model
from transformations import get_transformer
from pathlib import Path
import numpy as np
from PIL import Image



"""
This file takes an experiment that is given through the outputfolder, and creates predictions on the generalization set.
Then it makes predictions on the unlabeled data.
Since through cross validation 5 models were generated, it does that five times, but only the first one counts 
(i.e. unlabeled_predictions_1.csv and generalization_predictions_1.csv)
Note that to train the model on the whole dataset and get the best predictions on the unlabeled data, the --phase "complete_data"
has to be set.
"""
#only adjust this line
output_folder = Path("../Results/Complete_RIM_Oneshot_Tuned61_Weight100/")

resolution=256
folder = output_folder.joinpath("Config_1/")
data = pd.read_csv("../labels_complete.csv")
unlabeled_data=data[np.isnan(data["Label"])]

input_data = pd.read_csv(output_folder.joinpath("inputs_and_results.csv"))
task = input_data["task"][0]
transformation_type = input_data["transformation"][0]
oversampling_rate = input_data["oversampling_rate"][0]
transformation = get_transformer(transformation_type, 256)
model_name = input_data["model"][0]
data_payload = {"task": task, "phase": "test", "set": "train", "transform": transformation,
                "oversamplingrate": oversampling_rate, "split": 0, "resolution": resolution,
                "model_name": "resnet"}
model = get_model(model_name)
file_path = Path(f"../Images_{str(resolution)}")


loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
val_set = load_dataset(**data_payload)
val_set.set = "val"
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)
if torch.cuda.is_available():
    device = torch.device("cuda")
else: device = torch.device("cpu")

loss_list = []
auc_list = []
acc_list =[]
_95_quant_list = []
_95_threshold = []
_975_quant_list = []
_975_threshold = []
_100_quant_list = []
_100_threshold = []

for i in range(1,6):
    model_path = folder.joinpath(f"model_{i}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    val_loss, val_auc, val_acc, labels_list, preds_list, names_list = val_epoch(model, test_loader=val_loader,
                                                                            loss_fn=loss_fn, device=device,
                                                                            return_lists=True)
    loss_list += [val_loss]
    auc_list += [val_auc]
    acc_list += [val_acc]

    frame = pd.DataFrame({"Name":names_list, "Label":labels_list, "Prediction":preds_list})
    frame.Prediction = [x.item() for x in frame.Prediction]
    frame.Label = [x.item() for x in frame.Label]
    frame.to_csv(output_folder.joinpath(f"generalization_predictions_{i}.csv"))

    sorted_frame = frame.sort_values(by='Prediction', ascending=False)
    sorted_frame = sorted_frame.reset_index()

    indices = [i for i in range(len(sorted_frame)) if sorted_frame.Label[i] == 1]
    quantiles = np.quantile(indices, [0.95, 0.975, 1])
    _95_quant_list += [quantiles[0] / len(sorted_frame)]
    _95_threshold += [sorted_frame.loc[round(quantiles[0])].Prediction]
    _975_quant_list += [quantiles[1] / len(sorted_frame)]
    _975_threshold += [sorted_frame.loc[round(quantiles[1])].Prediction]

    _100_quant_list += [quantiles[2] / len(sorted_frame)]
    _100_threshold += [sorted_frame.loc[round(quantiles[2])].Prediction]

    # predict on unlabeled data
    transformation = get_transformer("normalize", resolution=resolution)
    unlabeled_names = []
    unlabeled_preds = []
    for name in unlabeled_data["Name"]:
        img = Image.open(file_path.joinpath(name))
        img = transformation(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            out = model(img)
        one_pred = torch.softmax(out, 1)[0][1]
        unlabeled_names += [name]
        unlabeled_preds += [one_pred.item()]

    predictions = pd.DataFrame({"Name": unlabeled_names, "Prediction": unlabeled_preds})
    predictions = predictions.sort_values(by='Prediction', ascending=False)
    predictions.to_csv(output_folder.joinpath(f"unlabeled_predictions_{i}"))


test_evaluation_metrics = pd.DataFrame({"Loss": loss_list, "AUC": auc_list, "ACC": acc_list, "95%":_95_quant_list,
                                        "97.5%":_975_quant_list, "100%":_100_quant_list,
                                        "95_threshold":_95_threshold,
                                        "975_threshold":_975_threshold,
                                        "100_threshold":_100_threshold})
test_evaluation_metrics.to_csv(output_folder.joinpath("test_metrics.csv"))

