from sklearn.metrics import confusion_matrix, roc_auc_score
import torch

def compute_measures(labels_list, preds_list):
    auc = roc_auc_score(labels_list, preds_list)
    tn, fp, fn, tp = confusion_matrix(labels_list, preds_list).ravel()
    acc = (tp + tn) / (tn + tp + fp + fn)
    return auc, acc


def train_epoch(model, train_loader,optimizer, loss_fn, device=torch.device("cpu")):
    model.train()  # Set model to training mode
    train_loss = 0
    preds_list = []
    labels_list = []

    for data in train_loader:
        inputs = data["Image"]
        labels = data["Label"]

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() # maybe * inputs.size(0) to control for uneven batches, but I think the lossfunction does that on its own
        _, preds = torch.max(output.data, 1)
        preds_list = preds_list + list(preds.cpu())
        labels_list = labels_list + list(labels.cpu())
    train_auc, train_acc = compute_measures(labels_list, preds_list)

    return train_loss, train_auc, train_acc

def val_epoch(model, test_loader, loss_fn, device=torch.device("cpu")):
    model.eval()  # Set model to training mode
    val_loss = 0
    preds_list = []
    labels_list = []

    for data in test_loader:
        inputs = data["Image"]
        labels = data["Label"]

        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss=loss_fn(output,labels)
        val_loss+=loss.item()
        _, preds = torch.max(output.data, 1)
        preds_list = preds_list + list(preds.cpu())
        labels_list = labels_list + list(labels.cpu())
    val_auc, val_acc = compute_measures(labels_list, preds_list)

    return val_loss, val_auc, val_acc
