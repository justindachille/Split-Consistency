import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils.calculate_acc import calculate_accuracy

def train_client_v2(net_id, net_client, train_dataloader, epochs, lr, args_optimizer, optimizer_server, args, round, shared_server_model, device, logger):
    print(f'training client {net_id} in sflv2')
    net_client = net_client.to(device)
    shared_server_model = shared_server_model.to(device)

    # Initialize client optimizer
    optimizer_client = get_optimizer(net_client, lr, args_optimizer, args)

    logger.info(f'Training network {net_id}')
    logger.info(f'n_training: {len(train_dataloader)}')

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer_client.zero_grad()

            # Forward pass on client model
            client_output = net_client(x)
            client_fx = client_output.clone().detach().requires_grad_(True)

            # Send client_output to shared server and get gradients
            dfx, loss, acc = train_server(client_fx, target, args, device, shared_server_model, optimizer_server)

            # Backward pass using gradients from server
            client_output.backward(dfx)
            optimizer_client.step()

            epoch_loss_collector.append(loss)
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info(f'Client: {net_id} Epoch: {epoch} Loss: {epoch_loss}')

    net_client.to('cpu')
    logger.info(f' ** Client: {net_id} Training complete **')

    return net_client.state_dict()

def train_server(client_output, target, args, device, shared_server_model, optimizer_server):
    # shared_server_model = shared_server_model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer_server.zero_grad()
    server_output = shared_server_model(client_output)

    loss = criterion(server_output, target)
    loss.backward()

    acc = calculate_accuracy(server_output, target)

    dfx_client = client_output.grad.clone().detach()

    return dfx_client, loss.item(), acc

def get_optimizer(model, lr, args_optimizer, args):
    if args_optimizer == 'adam':
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise ValueError(f'Unsupported optimizer type: {args_optimizer}')