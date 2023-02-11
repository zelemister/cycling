from sklearn.metrics import confusion_matrix, roc_auc_score
import torch


def compute_measures(labels_list, preds_list):
    auc = roc_auc_score(labels_list, preds_list)
    pred_classes = [0 if x < 0.5 else 1 for x in preds_list]
    tn, fp, fn, tp = confusion_matrix(labels_list, pred_classes).ravel()
    acc = (tp + tn) / (tn + tp + fp + fn)
    return auc, acc


def train_epoch(model, train_loader, optimizer, loss_fn, device=torch.device("cpu")):
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
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # maybe * inputs.size(0) to control for uneven batches, but I think the lossfunction does that on its own
        # _, preds = torch.max(output.data, 1)
        preds = output.data.softmax(1).transpose(0, 1)[1]
        preds_list = preds_list + list(preds.cpu())
        labels_list = labels_list + list(labels.cpu())
    train_auc, train_acc = compute_measures(labels_list, preds_list)

    return train_loss, train_auc, train_acc


def train_epoch_rim(model, train_loader, optimizer, loss_fn, device=torch.device("cpu"), stages=1):
    train_loss = 0
    preds_list = []
    labels_list = []

    if stages == 1:
        # largely copy train_epoch, could rewrite, so duplicate code disappears
        model.train()
        for data in train_loader:
            inputs = data["Image"]
            labels = data["Label"]
            labels = rim_only_labels(labels)
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # _, preds = torch.max(output.data, 1)
            preds = output.data.softmax(1).transpose(0, 1)[1]
            preds_list = preds_list + list(preds.cpu())
            labels_list = labels_list + list(labels.cpu())
        train_auc, train_acc = compute_measures(labels_list, preds_list)

        return train_loss, train_auc, train_acc

    elif stages == 2:
        cut_off = 0.1
        head_less = torch.nn.Sequential(*(list(model[0].children())[:-1]))
        rim_model = torch.nn.Sequential(head_less, model[1])
        model[0].train()
        rim_model.train()

        for data in train_loader:
            inputs = data["Image"]
            labels = data["Label"]
            labels_phase1 = torch.tensor([1 if x == 2 else x for x in labels])
            inputs, labels_phase1 = inputs.to(device), labels_phase1.to(device)
            optimizer[0].zero_grad()
            output = model[0](inputs)
            loss = loss_fn(output, labels_phase1)
            loss.backward()
            optimizer[0].step()
            preds_bikes = output.data.softmax(1).transpose(0, 1)[1]
            """
            inputs = inputs.cpu()
            rim_labels = rim_only_labels(labels)
            labels_phase2_filtered = [rim_labels[i] for i in range(len(preds_bikes)) if preds_bikes[i] >= cut_off]
            inputs_filtered = [inputs[i] for i in range(len(preds_bikes)) if preds_bikes[i] >= cut_off]
            labels_phase2_filtered = torch.tensor(labels_phase2_filtered)
            inputs_filtered = inputs_filtered.to(device)
            labels_phase2_filtered=labels_phase2_filtered.to(device)
            """
            rim_labels = rim_only_labels(labels)
            indizes = preds_bikes >= cut_off
            labels_phase2_filtered = labels[indizes].to(device)
            inputs_filtered = inputs[indizes].to(device)


            optimizer[1].zero_grad()
            output = rim_model(inputs_filtered)
            loss = loss_fn(output, labels_phase2_filtered)
            loss.backward()
            optimizer[1].step()
            preds_rim = output.data.softmax(1).transpose(0, 1)[1]
            preds = stitch_predictions(preds_bikes, preds_rim, cut_off)
            preds_list = preds_list + preds
            labels_list = labels_list + list(rim_labels)
            train_loss = None
        train_auc, train_acc = compute_measures(labels_list, preds_list)

        return train_loss, train_auc, train_acc

def val_epoch(model, test_loader, loss_fn, device=torch.device("cpu"), return_lists=False):
    model.eval()  # Set model to training mode
    val_loss = 0
    preds_list = []
    labels_list = []
    names_list = []
    for data in test_loader:
        inputs = data["Image"]
        labels = data["Label"]
        names = data["Name"]
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = loss_fn(output, labels)
        val_loss += loss.item()
        # _, preds = torch.max(output.data, 1)
        preds = output.data.softmax(1).transpose(0, 1)[1]
        preds_list = preds_list + list(preds.cpu())
        labels_list = labels_list + list(labels.cpu())
        names_list = names_list + names
    val_auc, val_acc = compute_measures(labels_list, preds_list)
    if return_lists:
        return val_loss, val_auc, val_acc, labels_list, preds_list, names_list
    return val_loss, val_auc, val_acc


def val_epoch_rim(model, test_loader, loss_fn, stages=1, device=torch.device("cpu")):
    val_loss = 0
    preds_list = []
    labels_list = []

    if stages == 1:
        model.eval()
    elif stages == 2:
        head_less = torch.nn.Sequential(*(list(model[0].children())[:-1]))
        rim_model = torch.nn.Sequential(head_less, model[1])
        rim_model.eval()

    for data in test_loader:
        inputs, labels = data["Image"], data["Label"]

        if stages == 1:
            labels = rim_only_labels(labels)
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
            output = model(inputs)
            loss = loss_fn(output, labels)
            val_loss += loss.item()
            preds = output.data.softmax(1).transpose(0, 1)[1]
            preds_list = preds_list + list(preds.cpu())
            labels_list = labels_list + list(labels.cpu())

        elif stages == 2:
            # Phase 1: predict on bike_lanes with model[0]
            cut_off = 0.05
            labels_bikes = [1 if x == 2 else x for x in labels]
            inputs, labels_bikes = inputs.to(device), labels_bikes.to(device)
            output = model[0](inputs)
            preds_bikes = output.data.softmax(1).transpose(0, 1)[1]


            # Phase 2: Predict those labels with high enough probability with rim_model
            inputs, labels = inputs.cpu(), labels.cpu()
            labels_phase2_filtered = [labels[i] for i in range(len(preds_bikes)) if preds_bikes[i] >= cut_off]
            inputs_filtered = [inputs[i] for i in range(len(preds_bikes)) if preds_bikes[i] >= cut_off]
            labels_phase2_filtered = torch.tensor(rim_only_labels(labels_phase2_filtered))
            rim_labels = rim_only_labels(labels)
            inputs_filtered,labels_phase2_filtered = inputs_filtered.to(device), labels_phase2_filtered.to(device)

            output = rim_model(inputs_filtered)
            preds_rim = output.data.softmax(1).transpose(0, 1)[1]

            preds = stitch_predictions(preds_bikes, preds_rim, cut_off)
            preds_list = preds_list + preds
            labels_list = labels_list + list(rim_labels)
            val_loss = None


    val_auc, val_acc = compute_measures(labels_list, preds_list)

    return val_loss, val_auc, val_acc


def rim_only_labels(labels):
    """
    This function collapses the labels of our dataset (0,1,2) into (0,1) by setting the bikelanes to zero, and then the
    RIMs to 1.
    :param labels: input labels with 3 possible labels (0,1,2)
    :return: labels with 2  labels (0,1) (no_rim, rim)
    """
    labels = [0 if x == 1 else x for x in labels]
    labels = [1 if x == 2 else x for x in labels]
    return labels

def stitch_predictions(preds_bikelane, preds_rim, cut_off:float):
    """
    This function creates a list of predicted labels of length (len(preds_bikelane)) where all the values,
    which didn't exceed a certain threshold got set to zero, and all others were replaced with the labels from preds_rim
    :param preds_bikelane: tensor of predictions from the first phase
    :param preds_rim: tensor of predictions from the second phase
    :param cut_off: cut_off point
    :return:
    """
    preds_rim = list(preds_rim.cpu())
    preds_bikelane = list(preds_bikelane.cpu())
    preds = [0]*len(preds_bikelane)
    indizes = [i for i in range(len(preds_bikelane)) if preds_bikelane[i] >= cut_off]
    preds_rim.reverse()
    preds = [preds_rim.pop() if i in indizes else preds[i] for i in range(len(preds))]
    return preds