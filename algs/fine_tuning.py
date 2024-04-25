import time 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils.calculate_acc import compute_accuracy

def train_net(net, train_dl_local, optimizer, device, args):
    net = net.to(device)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl_local):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, _, out, _, _ = net(inputs)
        loss = F.cross_entropy(out, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = out.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = correct / total
    train_loss = train_loss / len(train_dl_local)

    return train_acc, train_loss

def fine_tune(net, train_dl_local, test_dl_local, test_dl_global, device, args, logger, max_patience=3):
    print('dev', device)
    patience = 0
    best_acc = 0.0
    best_local_acc = 0.0
    best_local_acc_top5 = 0.0
    best_global_acc = 0.0
    best_global_acc_top5 = 0.0
    
    net = net.to(device)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg, amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    for epoch in range(args.comm_round):
        train_acc, train_loss = train_net(net, train_dl_local, optimizer, device, args)
        local_test_acc, _, local_test_acc_top5 = compute_accuracy(net, test_dl_local, get_confusion_matrix=False, device=device)
        global_test_acc, _, global_test_acc_top5 = compute_accuracy(net, test_dl_global, get_confusion_matrix=False, device=device)

        print(f'Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Local Test Acc: {local_test_acc:.4f} | Global Test Acc: {global_test_acc:.4f}')

        best_local_acc = max(local_test_acc, best_local_acc)
        best_local_acc_top5 = max(local_test_acc_top5, best_local_acc_top5)
        best_global_acc = max(global_test_acc, best_global_acc)
        best_global_acc_top5 = max(global_test_acc_top5, best_global_acc_top5)

        if local_test_acc > best_acc:
            best_acc = local_test_acc
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break

    return best_local_acc, best_local_acc_top5, best_global_acc, best_global_acc_top5