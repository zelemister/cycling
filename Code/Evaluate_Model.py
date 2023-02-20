import pandas as pd
import torch.utils.data

from training import val_epoch
from DatasetGenerator import load_dataset
from model_loaders import get_model
from transformations import get_transformer
from pathlib import Path
import numpy as np
from PIL import Image
#model_path = Path("../Results/RIM_Oneshot_Tuned61/Config_1/trained_model.pt")
resolution=256
model_path = Path("../Results/Bikelane_tunedFor2Phase/Config_1/model_1.pt")
data = pd.read_csv("../labels_complete.csv")
unlabeled_data=data[np.isnan(data["Label"])]
transformation = get_transformer("rotations", 256)
data_payload = {"task": "bikelane", "phase": "test", "set": "train", "transform": transformation,
                "oversamplingrate": 43, "split": 0, "resolution": resolution,
                "model_name": "resnet"}
#model = get_model("resnet18")
model = get_model("resnet34")
file_path = Path(f"../Images_{str(resolution)}")


model.eval()
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
val_set = load_dataset(**data_payload)
val_set.set = "val"
val_loader = torch.utils.data.DataLoader(val_set, batch_size=12, shuffle=False)
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

indices = [i for i in range(len(sorted_frame)) if sorted_frame.Label[i] == 1]
print(sorted_frame.head(max(indices) + 1).Label.value_counts())



"""#predict on unlabeled data
transformation = get_transformer("normalize", resolution=resolution)
unlabeled_names = []
unlabeled_preds = []
for name in unlabeled_data["Name"]:
    img = Image.open(file_path.joinpath(name))
    img = transformation(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        out = model(img)
    one_pred = torch.softmax(out, 1)[0][1]
    unlabeled_names += [name]
    unlabeled_preds += [one_pred]

predictions = pd.DataFrame({"Name":unlabeled_names, "Prediction":unlabeled_preds})
predictions = predictions.sort_values(by='Prediction', ascending=False)
print(predictions)
"""