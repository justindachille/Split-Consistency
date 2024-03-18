from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch 
import numpy as np

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):
    model.cuda()
    model.eval()

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    correct, total = 0, 0
    top5_correct = 0

    criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            _, _, out, _, _ = model(x)

            loss = criterion(out, target)
            loss_collector.append(loss.item())

            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            # Calculate top-5 accuracy
            _, top5_pred = out.topk(5, 1, True, True)
            top5_correct += (top5_pred == target.data.view(-1, 1).expand_as(top5_pred)).sum().item()

            pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
            true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    avg_loss = sum(loss_collector) / len(loss_collector)
    top1_accuracy = correct / float(total)
    top5_accuracy = top5_correct / float(total)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
        return top1_accuracy, conf_matrix, avg_loss, top5_accuracy

    model.train()
    return top1_accuracy, avg_loss, top5_accuracy

def compute_accuracy_split_model(global_net_client, global_net_server, dataloader, get_confusion_matrix=False, device="cpu"):
    global_net_client.to(device).eval()
    global_net_server.to(device).eval()

    true_labels_list, pred_labels_list = np.array([]), np.array([])
    correct, total = 0, 0
    top5_correct = 0

    criterion = nn.CrossEntropyLoss().to(device)
    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)

            client_output = global_net_client(x)
            server_output = global_net_server(client_output)

            loss = criterion(server_output, target)
            loss_collector.append(loss.item())

            _, pred_label = torch.max(server_output.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            _, top5_pred = server_output.topk(5, 1, True, True)
            top5_correct += (top5_pred == target.data.view(-1, 1).expand_as(top5_pred)).sum().item()

            pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
            true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    avg_loss = sum(loss_collector) / len(loss_collector)
    top1_accuracy = correct / float(total)
    top5_accuracy = top5_correct / float(total)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
        return top1_accuracy, conf_matrix, avg_loss, top5_accuracy

    global_net_client.train()
    global_net_server.train()

    return top1_accuracy, avg_loss, top5_accuracy

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc
