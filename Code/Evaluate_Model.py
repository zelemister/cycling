import pandas as pd
import torch.utils.data

from training import val_epoch
from DatasetGenerator import load_dataset
from model_loaders import get_model
from transformations import get_transformer
from pathlib import Path

model_path = Path("../Results/RIM_Oneshot_Tuned61/Config_1/trained_model.pt")
unlabeled_data = pd.read_csv("../labels_complete.csv")
transformation = get_transformer("rotations", 256)
data_payload = {"task": "one_shot", "phase": "test", "set": "train", "transform": transformation,
                "oversamplingrate": 43, "split": 0, "resolution": 256,
                "model_name": "resnet"}
model = get_model("resnet18")
model.eval()
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
val_set = load_dataset(**data_payload)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=12)
labels_list = []
names_list = []
if torch.cuda.is_available():
    device = torch.device("cuda")
else: device = torch.device("cpu")
model.load_state_dict(torch.load(model_path, map_location=device))

val_loss, val_auc, val_acc, labels_list, preds_list, names_list = val_epoch(model, test_loader=val_loader,
                                                                            loss_fn=loss_fn, device=device,
                                                                            return_lists=True)
print(round(val_loss,3))
print(round(val_auc,3))
print(round(val_acc,3))
frame = pd.DataFrame({"Name":names_list, "Label":labels_list, "Prediction":preds_list})
frame.Prediction = [x.item() for x in frame.Prediction]
frame.Label = [x.item() for x in frame.Label]
sorted_frame = frame.sort_values(by='Prediction', ascending=False)
sorted_frame = sorted_frame.reset_index()
print(frame)

indezes = [i for i in range(len(sorted_frame)) if sorted_frame.Label[i]==1]
print(sorted_frame.head(max(indezes)+1).Label.value_counts())