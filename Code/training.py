from sklearn.metrics import confusion_matrix, roc_auc_score
import torch

def compute_measures(labels_list, preds_list):
    auc = roc_auc_score(labels_list, preds_list)
    pred_classes = [0 if x < 0.5 else 1 for x in preds_list]
    tn, fp, fn, tp = confusion_matrix(labels_list, pred_classes).ravel()
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
        #_, preds = torch.max(output.data, 1)
        preds = output.data.softmax(1).transpose(0,1)[1]
        preds_list = preds_list + list(preds.cpu())
        labels_list = labels_list + list(labels.cpu())
    train_auc, train_acc = compute_measures(labels_list, preds_list)

    return train_loss, train_auc, train_acc

def train_epoch_rim(model, train_loader, optimizer, loss_fn, device = torch.device("cpu"),phases = 1):
    train_loss = 0
    preds_list = []
    labels_list = []

    if phases == 1:
        # largely copy train_epoch, could rewrite, so duplicate code disappears
        for data in train_loader:
            model.train()
            inputs = data["Image"]
            labels = data["Label"]
            labels = [0 if x==1 else x for x in labels]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # maybe * inputs.size(0) to control for uneven batches, but I think the lossfunction does that on its own
            #_, preds = torch.max(output.data, 1)
            preds = output.data.softmax(1).transpose(0, 1)[1]
            preds_list = preds_list + list(preds.cpu())
            labels_list = labels_list + list(labels.cpu())
        train_auc, train_acc = compute_measures(labels_list, preds_list)

        return train_loss, train_auc, train_acc
    elif phases ==2:
        cutoff = 0.2
        for data in train_loader:
            model[0].train()
            inputs = data["Image"]
            labels = data["Label"]
            labels_phase1 = [1 if x == 2 else x for x in labels]
            inputs, labels_phase1 = inputs.to(device), labels_phase1.to(device)
            optimizer[0].zero_grad()
            output = model[0](inputs)
            loss = loss_fn(output, labels_phase1)
            loss.backward()
            optimizer[0].step()
            #_, preds_p1 = torch.max(output.data, 1)
            preds_p1 = output.data.softmax(1).transpose(0, 1)[1]

            inputs = inputs.cpu()
            labels_phase2 = [0 if x == 1 else x for x in labels_phase2]
            labels_phase2 = [1 if x == 2 else x for x in labels_phase2]
            labels_phase2_filtered= [labels[i] for i in range(len(preds_p1)) if preds_p1[i] >= cutoff]

            inputs_filtered= [inputs[i] for i in range(len(preds_p1)) if preds_p1[i]>= cutoff]
            inputs_filtered, labels_phase2_filtered = inputs_filtered.to(device), labels_phase2_filtered.to(device)

            optimizer[1].zero_grad()
            output = model[1](inputs_filtered)
            loss = loss_fn(output, labels_phase2_filtered)
            loss.backward()
            optimizer[1].step()
            #train_loss += loss.item()
            #_, preds_p2 = torch.max(output.data, 1)
            preds_p2 = output.data.softmax(1).transpose(0, 1)[1]

            preds = [0]*len(labels)
            indizes = [i for i in range(len(preds_p1)) if preds_p1[i] >= cutoff]
            preds = [preds_p2[i] if i in indizes else preds[i] for i in range(len(preds))]
            loss = loss_fn(preds, labels)

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
        #_, preds = torch.max(output.data, 1)
        preds = output.data.softmax(1).transpose(0,1)[1]
        preds_list = preds_list + list(preds.cpu())
        labels_list = labels_list + list(labels.cpu())
    val_auc, val_acc = compute_measures(labels_list, preds_list)

    return val_loss, val_auc, val_acc
