from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch 
import numpy as np

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    model.cuda()
    model.eval()
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0

    #criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            #x, target = x.to(device), target.to(dtype=torch.int64).to(device)
            #x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)

            _, _, out, _, _ = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())
            _, pred_label = torch.max(out.data, 1)
            
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()


            pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
            true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss
    
    model.train()
    
    return correct / float(total), avg_loss

def compute_accuracy_split_model(global_net_client, global_net_server, dataloader, get_confusion_matrix=False, device="cpu"):
    global_net_client.to(device).eval()
    global_net_server.to(device).eval()
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
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

            pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
            true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)
        return correct / float(total), conf_matrix, avg_loss

    global_net_client.train()
    global_net_server.train()

    return correct / float(total), avg_loss

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc
