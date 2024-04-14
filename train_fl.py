import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import csv
from importlib import reload

import argparse

from torch.autograd import Variable

import time
import torch.nn.functional as F
import numpy as np
import copy

from utils.dataloader import DatasetSplit, partition_data, get_dataloader, FedAvg

from nets.models import SimpleCNN
from nets.models import AlexNet, AlexNetClient, AlexNetServer
from nets.models import ResNet_18, ResidualBlock, ResNet_18_client_side, ResNet_18_server_side
from nets.models import ResNet_50, ResNet_50_client_side, ResNet_50_server_side
from nets.models import VGG16, VGG16_client_side, VGG16_server_side

from algs.fedavg import train_net_fedavg
from algs.fedprox import train_net_fedprox
from algs.moon import train_net_moon
from algs.feduv import train_net_feduv
from algs.sflv1 import train_client_v1
from algs.sflv2 import train_client_v2
from algs.fine_tuning import fine_tune

from utils.calculate_acc import compute_accuracy, compute_accuracy_split_model

def print_split_parameter_settings(client_net, server_net, i):
    if i > 0:
        return
    total_params = sum(p.numel() for p in client_net.parameters()) + sum(p.numel() for p in server_net.parameters())

    # Print the number of parameters and percentage for each model
    print("Client-side model:")
    client_params = sum(p.numel() for p in client_net.parameters())
    print(f"Number of parameters: {client_params}")
    print(f"Percentage of total parameters: {client_params / total_params * 100:.2f}%")

    print("Server-side model:")
    server_params = sum(p.numel() for p in server_net.parameters())
    print(f"Number of parameters: {server_params}")
    print(f"Percentage of total parameters: {server_params / total_params * 100:.2f}%\n")
    

def init_nets(n_parties, args, device, n_classes):
    nets = {net_i: (None, None) for net_i in range(n_parties)}
    global_server_model = None
    if args.alg == 'sflv2':
        if args.model == 'alexnet':
            global_server_model = AlexNetServer(args, n_classes)
        elif args.model == 'resnet-18':
            global_server_model = ResNet_18_server_side(ResidualBlock, args, num_classes=n_classes)
        elif args.model == 'resnet-50':
            global_server_model = ResNet_50_server_side(args, num_classes=n_classes)
        elif args.model == 'vgg-16':
            global_server_model = VGG16_server_side(args, num_classes=n_classes)
            
    for net_i in range(n_parties):
        if args.model == 'resnet-50':
            if args.alg == 'sflv1':
                client_net = ResNet_50_client_side(args)
                server_net = ResNet_50_server_side(args, num_classes=n_classes)
                print_split_parameter_settings(client_net, server_net, net_i)
            elif args.alg == 'sflv2':
                client_net = ResNet_50_client_side(args)
                server_net = None
            else:
                net = ResNet_50(args, n_classes)
        elif args.model == 'resnet-18':
            if args.alg == 'sflv1':
                client_net = ResNet_18_client_side(ResidualBlock, args)
                server_net = ResNet_18_server_side(ResidualBlock, args, num_classes=n_classes)
                print_split_parameter_settings(client_net, server_net, net_i)
            elif args.alg == 'sflv2':
                client_net = ResNet_18_client_side(ResidualBlock, args)
                server_net = None
            else:
                net = ResNet_18(args, n_classes)
        elif args.model == 'simple-cnn':
            net = SimpleCNN(args.out_dim, n_classes, args.simp_width)
        elif args.model == 'vgg-16':
            if args.alg == 'sflv1':
                client_net = VGG16_client_side(args)
                server_net = VGG16_server_side(args, num_classes=n_classes)
                print_split_parameter_settings(client_net, server_net, net_i)
            elif args.alg == 'sflv2':
                client_net = VGG16_client_side(args)
                server_net = None
            else:
                net = VGG16(n_classes)
        if args.alg == 'sflv1':
            nets[net_i] = (client_net.to(device), server_net.to(device))
        elif args.alg == 'sflv2':
            nets[net_i] = (client_net.to(device), None)
        else:
            nets[net_i] = net.to(device)

    if args.alg == 'sflv1' or args.alg == 'sflv2':
        model_meta_data = []
        layer_type = []
        for (k, v) in nets[0][0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)
    else:
        model_meta_data = []
        layer_type = []
        for (k, v) in nets[0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

    return nets, model_meta_data, layer_type, global_server_model

def main(args):
    #0. Set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)    
    
    #1. check device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cpu':
        device = torch.device('cpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device != '':
        cuda, number = args.device.split(":")
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{number}"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if type(device) != str and args.device == 'cuda' and device.type == 'cpu':
        print("GPU not detected, defaulting to CPU")

    print("Device: ", device)

    if type(device) != str and device.type == 'cuda':
        args.multiprocessing=0
        
    print(f"Algorithm: {args.alg}")

    #2. Init logs
    hparams = {
        'alg': args.alg,
        'partition': args.partition,
        'lr': args.lr,
        'epochs': args.epochs,
        'num_users': args.n_parties,
        # 'split_layer': args.split_layer,
        'beta': args.alpha,
        'batch_size': args.batch_size,
        'opt': args.optimizer,
        'dataset': args.dataset,
    }
    hparam_str = "_".join(f"{k}={v}" for k, v in hparams.items())
    if args.log_file_name is None:
        args.log_file_name = f'logs/experiment_log-{hparam_str}_{datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")}'
    log_path = args.log_file_name + '.log'

    reload(logging)

    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    args_string = json.dumps(vars(args), indent=4)
    print(args_string)
    logging.info("Command Line Arguments: \n%s", args_string)
    #3. Get data
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)
            
    if args.dataset == 'ham10000':
        def print_aggregated_labels(dataloader, client_name):
            label_counts = {}
            for batch in dataloader:
                labels = batch[1]
                for label in labels:
                    label = label.item()
                    if label in label_counts:
                        label_counts[label] += 1
                    else:
                        label_counts[label] = 1

            print(f"Aggregated label counts for {client_name}: {label_counts}")
        train_dl_global, test_dl, client1_loader, client2_loader, net_dataidx_map = partition_data(
            args,
            args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.alpha, 
           logger=logger
        )
        test_ds_global = []
        train_ds_global = []
        train_dl_local_list, test_dl_local_list = [],[]
        print_aggregated_labels(client1_loader, 'client1')
        print_aggregated_labels(client2_loader, 'client2')
        train_dl_local_list.append(client1_loader)
        train_dl_local_list.append(client2_loader)

        test_dl_local_list.append(test_dl)
        test_dl_local_list.append(test_dl)
        n_classes = 7

    elif args.dataset == 'stl10'or args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
            args,
            args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.alpha, 
           logger=logger
        )
        
        n_classes = len(np.unique(y_train))
        train_dl_global, test_dl, train_ds_global, test_ds_global, _ = get_dataloader(args, args.dataset,
                                                                                   args.datadir,
                                                                                   args.batch_size,
                                                                                   args.batch_size,
                                                                                   X_train, y_train,
                                                                                   X_test, y_test)

        train_dl_local_list, test_dl_local_list = [],[]
        for i in range(args.n_parties):
            dataidxs = net_dataidx_map[i]
            train_dl_local, _, _, _, test_dl_local = get_dataloader(args, args.dataset, 
                                                                 args.datadir, 
                                                                 args.batch_size, 
                                                                 args.batch_size, 
                                                                 X_train, y_train,
                                                                 X_test, y_test,
                                                                 dataidxs, partition=args.partition)
            
            train_dl_local_list.append(train_dl_local)
            test_dl_local_list.append(test_dl_local)

    else:
        all_train_ds, train_ds_global, train_dl_global, test_ds_global, test_dl = load_feature_shift(args)
        
        n_classes = len(np.unique(test_ds_global.samples[:,1]))

        all_train_ds = []
        for domain in args.domains:
            all_train_ds.append(dl_obj(f"{args.datadir}/{args.dataset}/train/{domain}", transform=transform_train))

        train_ds_global = torch.utils.data.ConcatDataset(all_train_ds)

        train_dl_global = data.DataLoader(dataset=train_ds_global, batch_size=args.batch_size, num_workers=8, shuffle=True, 
                                 pin_memory=True, persistent_workers=True
                                 )


        test_ds_global = dl_obj(f"{args.datadir}/{args.dataset}/test", transform=transform_test)
        test_dl = data.DataLoader(dataset=test_ds_global, batch_size=args.batch_size, num_workers=4, shuffle=False, 
                                 pin_memory=True, persistent_workers=True
                                 )
        
    train_dl=None
    data_size = len(test_ds_global)

    if args.alg != 'sflv2':
        nets, local_model_meta_data, layer_type, _ = init_nets(args.n_parties, args, device, n_classes)
        global_models, global_model_meta_data, global_layer_type, _ = init_nets(1, args, device, n_classes)
        global_model = global_models[0]
    else:
        nets, local_model_meta_data, layer_type, global_server_model = init_nets(args.n_parties, args, device, n_classes)
        global_models, global_model_meta_data, global_layer_type, _ = init_nets(1, args, device, n_classes)
        global_model = global_models[0]

    n_comm_rounds = args.comm_round

    if args.alg == 'moon':
        old_nets_pool = []

        if args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

    n_epoch = args.epochs
    if args.alg == 'sflv1':
        global_client_model, global_server_model = global_model
    elif args.alg == 'sflv2':
        global_client_model, _ = global_model

    if args.alg == 'sflv1' or args.alg == 'sflv2':
        global_client_optimizer = optim.SGD(global_client_model.parameters(), lr=args.lr, momentum=0.9,
                                            weight_decay=args.reg)
        global_server_optimizer = optim.SGD(global_server_model.parameters(), lr=args.lr, momentum=0.9,
                                            weight_decay=args.reg)

        global_optimizer = (global_client_optimizer, global_server_optimizer)

        client_scheduler = optim.lr_scheduler.CosineAnnealingLR(global_client_optimizer, n_comm_rounds)
        server_scheduler = optim.lr_scheduler.CosineAnnealingLR(global_server_optimizer, n_comm_rounds)

        scheduler = (client_scheduler, server_scheduler)
    else:
        global_optimizer = optim.SGD(global_model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.reg)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(global_optimizer, n_comm_rounds)

    best_global_train = 0.0
    best_global_test = 0.0
    
    best_global_train_top5 = 0.0
    best_global_test_top5 = 0.0
    for round in range(n_comm_rounds):
        if args.alg == 'local_training':
            break
        
        print("\n\n************************************")
        print("round: ", round)
        cur_time = time.time()
        logger.info("in comm round:" + str(round))

        if isinstance(scheduler, tuple):
            cur_lr = scheduler[0].get_last_lr()[0]
        else:
            cur_lr = scheduler.get_last_lr()[0]
        print(f"Current LR: {cur_lr}")

        party_list_this_round = party_list_rounds[round]

        # Check if the global model is a tuple (for sflv1)
        if args.alg == 'sflv1':
            global_client_model, global_server_model = global_model
            global_client_model.eval()
            for param in global_client_model.parameters():
                param.requires_grad = False
            global_client_w = global_client_model.state_dict()
            global_server_w = global_server_model.state_dict()

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net_id, (client_net, server_net) in nets_this_round.items():
                client_net.load_state_dict(global_client_w)
                server_net.load_state_dict(global_server_w)
        elif args.alg == 'sflv2':
            global_client_model, _ = global_model
            global_client_model.eval()
            for param in global_client_model.parameters():
                param.requires_grad = False
            global_client_w = global_client_model.state_dict()

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net_id, (client_net, _) in nets_this_round.items():
                client_net.load_state_dict(global_client_w)
        else:
            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            # Load the global model state dict into each net in this round
            nets_this_round = {k: nets[k] for k in party_list_this_round}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)
        if args.alg == 'Freeze':
            print("Freezing Weights")
            for net in nets_this_round.values():
                for param in net.fc3.parameters():
                    param.requires_grad = False

        avg_acc = 0.0
        acc_list = []
        if global_model:
            if args.alg == 'sflv1':
                global_model[0].to(device)
                global_model[1].to(device)
            elif args.alg == 'sflv2':
                global_model[0].to(device)
                global_server_model.to(device)
            else:
                global_model.to(device)

        local_weights = []

        procs=[]
        w_locals = []
        c_locals = [] # Only used for scaffold
        c_deltas = [] # Only used for scaffold
        w_locals_client = [] # Only used for split
        w_locals_server = [] # Only used for split
        for net_id, net in nets_this_round.items():
            train_dl_local, test_dl_local = train_dl_local_list[net_id], test_dl_local_list[net_id]
            logger.info(f"Training network {(str(net_id))} n_training: {len(train_dl_local.dataset)}")

            if len(train_dl_local.dataset) == 0:
                logger.info(f"Skipping training for network {net_id} due to no data")
                continue
                
            if args.alg == 'moon':
                prev_models=[]
                for i in range(len(old_nets_pool)):
                    prev_models.append(old_nets_pool[i][net_id])

                local = train_net_moon(net_id, net, global_model, prev_models, 
                                              train_dl_local, test_dl, n_epoch, cur_lr,
                                              args.optimizer, args.mu, args.temperature, 
                                              args, round, device, logger)
                w_locals.append(local)            

            elif args.alg == 'fedprox':
                single_local = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, cur_lr,
                                                              args.optimizer, args, round, 
                                                             device, logger)

                w_locals.append(single_local)

            elif args.alg == 'fedavg' or args.alg == 'freeze':
                single_local = train_net_fedavg(net_id, net, train_dl_local, test_dl, n_epoch, cur_lr,
                                                              args.optimizer, args, round, 
                                                            device, logger)

                w_locals.append(single_local)

            elif args.alg == 'feduv':
                single_local = train_net_feduv(net_id, net, train_dl_local, test_dl, n_classes, n_epoch, cur_lr, 
                                  args.optimizer, args, round, device, logger)

                w_locals.append(single_local)
            
            elif args.alg == 'sflv1':
                client_server_nets = nets[net_id]
                client_model_state, server_model_state = train_client_v1(net_id, client_server_nets, train_dl_local, n_epoch, cur_lr,
                                                                        args.optimizer, args, round, device, logger)
                
                w_locals_client.append(client_model_state)
                w_locals_server.append(server_model_state)
            
            elif args.alg == 'sflv2':
                client_net = nets[net_id][0]
                client_model_state = train_client_v2(net_id, client_net, train_dl_local, n_epoch, cur_lr,
                                                    args.optimizer, global_server_optimizer, args, round, 
                                                    global_server_model, device, logger)
                
                w_locals_client.append(client_model_state)

        avg_acc /= args.n_parties
        if args.alg == 'local_training':
            logger.info("avg test acc %f" % avg_acc)
            logger.info("std acc %f" % np.std(acc_list))

        if args.alg == "sflv1":
            global_client_w = None
            global_server_w = None
        else:
            global_w = None

        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

        layer_exclude_list = []
        if args.alg == 'freeze':
            layer_exclude_list.append("fc3")

        print("TIME: ", time.time() - cur_time)

        logger.info('global n_training: %d' % len(train_dl_global))
        logger.info('global n_test: %d' % len(test_dl))

        if args.alg == "sflv1":
            w_glob_client = FedAvg(w_locals_client)
            w_glob_server = FedAvg(w_locals_server)

            global_client_model, global_server_model = global_model
            global_client_model.load_state_dict(w_glob_client)
            global_server_model.load_state_dict(w_glob_server)

            client_scheduler, server_scheduler = scheduler
            client_scheduler.step()
            server_scheduler.step()

            global_model = (global_client_model, global_server_model)
            
            global_model[0].to(device)
            global_model[1].to(device)

            train_acc, train_loss, train_acc_top5 = compute_accuracy_split_model(global_client_model, global_server_model, train_dl_global, device=device)
            test_acc, _, test_acc_top5 = compute_accuracy_split_model(global_client_model, global_server_model, test_dl, get_confusion_matrix=False, device=device)
            global_model[0].to('cpu')
            global_model[1].to('cpu')
        elif args.alg == "sflv2":
            w_glob_client = FedAvg(w_locals_client)

            global_client_model, _ = global_model
            global_client_model.load_state_dict(w_glob_client)

            client_scheduler, server_scheduler = scheduler
            client_scheduler.step()
            server_scheduler.step()

            global_model[0].to(device)
            global_server_model.to(device)
            global_client_model, _ = global_model

            train_acc, train_loss, train_acc_top5 = compute_accuracy_split_model(global_client_model, global_server_model, train_dl_global, device=device)
            test_acc, _, test_acc_top5 = compute_accuracy_split_model(global_client_model, global_server_model, test_dl, get_confusion_matrix=False, device=device)
            global_model[0].to('cpu')
            global_server_model.to('cpu')
        else:
            w_glob_model = FedAvg(w_locals)
            global_model.load_state_dict(w_glob_model)
            scheduler.step()
            global_model.to(device)
            train_acc, train_loss, train_acc_top5 = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, _, test_acc_top5 = compute_accuracy(global_model, test_dl, get_confusion_matrix=False, device=device)
            global_model.to('cpu')

        logger.info(f'>> Global Model Train accuracy: {train_acc:.4f}, Train accuracy top-5: {train_acc_top5:.4f}')
        logger.info(f'>> Global Model Test accuracy: {test_acc:.4f}, Test accuracy top-5: {test_acc_top5:.4f}')
        logger.info(f'>> Global Model Train loss: {train_loss:.4f}')

        print(f'>> Global Model Train accuracy: {train_acc:.4f}, Train accuracy top-5: {train_acc_top5:.4f}')
        print(f'>> Global Model Test accuracy: {test_acc:.4f}, Test accuracy top-5: {test_acc_top5:.4f}')
        print(f'>> Global Model Train loss: {train_loss:.4f}')

        best_global_train = max(best_global_train, train_acc)
        best_global_test = max(best_global_test, test_acc)
        best_global_train_top5 = max(best_global_train_top5, train_acc_top5)
        best_global_test_top5 = max(best_global_test_top5, test_acc_top5)
        
        if args.alg == 'moon':
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size-2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i+1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

        print("TIME: ", time.time()-cur_time)
        print("************************************\n\n")

    def should_terminate(patience, max_patience, acc, best_acc):
        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
        
        should_stop = patience >= max_patience
        
        return should_stop, patience, best_acc,
    
    def get_best_accs(acc, acc_top5, best_acc, best_acc_top5):
        """
        Function to get the best top-1 and top-5 accuracies.

        Returns:
            tuple: (best_acc, best_acc_top5)
        """
        best_acc = max(acc, best_acc)
        best_acc_top5 = max(acc_top5, best_acc_top5)
        return best_acc, best_acc_top5    
    
    print("Starting Fine Tuning...")
    logger.info("Starting Fine Tuning...")

    best_local_accs = [0.0] * args.n_parties
    best_local_accs_top5 = [0.0] * args.n_parties
    best_global_accs = [0.0] * args.n_parties
    best_global_accs_top5 = [0.0] * args.n_parties

    if args.partition == 'subsample':
        for net_id in range(args.n_parties):
            print(f'Fine-tuning user {net_id}')
            train_dl_local, test_dl_local = train_dl_local_list[net_id], test_dl_local_list[net_id]
            if args.alg == 'sflv2':
                save_global_server_model = copy.deepcopy(global_server_model)

            if args.alg in ['fedavg', 'fedprox', 'moon', 'local_training']:
                net = nets[net_id]
                best_local_acc, best_local_acc_top5, best_global_acc, best_global_acc_top5 = fine_tune(net, train_dl_local, test_dl_local, test_dl, device, args, logger)
                best_local_accs[net_id] = best_local_acc
                best_local_accs_top5[net_id] = best_local_acc_top5
                best_global_accs[net_id] = best_global_acc
                best_global_accs_top5[net_id] = best_global_acc_top5
            elif args.alg == 'sflv1':
                patience = 0
                max_patience = 3
                best_acc = 0.0
                print(f'Fine tuning user {net_id}')
                client_server_nets = nets[net_id]
                for i in range(50):
                    train_dl_local = train_dl_local_list[net_id]
                    client_model_state, server_model_state = train_client_v1(net_id, client_server_nets, train_dl_local, n_epoch, cur_lr, args.optimizer, args, round, device, logger)

                    client_server_nets[0].load_state_dict(client_model_state)
                    client_server_nets[1].load_state_dict(server_model_state)

                    local_test_acc, _, local_test_acc_top5 = compute_accuracy_split_model(client_server_nets[0], client_server_nets[1], test_dl_local_list[net_id], get_confusion_matrix=False, device=device)
                    global_test_acc, _, global_test_acc_top5 = compute_accuracy_split_model(client_server_nets[0], client_server_nets[1], test_dl, get_confusion_matrix=False, device=device)

                    print(f'Local Acc: {local_test_acc} Global Acc: {global_test_acc}')

                    best_local_accs[net_id], best_local_accs_top5[net_id] = get_best_accs(local_test_acc, local_test_acc_top5, best_local_accs[net_id], best_local_accs_top5[net_id])
                    best_global_accs[net_id], best_global_accs_top5[net_id] = get_best_accs(global_test_acc, global_test_acc_top5, best_global_accs[net_id], best_global_accs_top5[net_id])

                    should_stop, patience, best_acc = should_terminate(patience, max_patience, local_test_acc, best_acc)

                    if should_stop:
                        print(f'Client {net_id} stopping after {i} epochs')
                        break
            elif args.alg == 'sflv2':
                for net_id in range(args.n_parties):
                    patience = 0
                    max_patience = 3
                    best_acc = 0.0
                    print(f'Fine tuning user {net_id}')
                    client_server_nets = nets[net_id]
                    global_server_model = copy.deepcopy(save_global_server_model)
                    for i in range(50):
                        train_dl_local = train_dl_local_list[net_id]

                        client_net = nets[net_id][0]
                        client_model_state = train_client_v2(net_id, client_net, train_dl_local, n_epoch, cur_lr,
                                                            args.optimizer, global_server_optimizer, args, round, 
                                                            global_server_model, device, logger)

                        w_locals_client.append(client_model_state)

                        client_server_nets[0].load_state_dict(client_model_state)

                        local_test_acc, _, local_test_acc_top5 = compute_accuracy_split_model(client_server_nets[0], global_server_model, test_dl_local_list[net_id], get_confusion_matrix=False, device=device)
                        global_test_acc, _, global_test_acc_top5 = compute_accuracy_split_model(client_server_nets[0], global_server_model, test_dl, get_confusion_matrix=False, device=device)

                        print(f'Local Acc: {local_test_acc} Global Acc: {global_test_acc}')

                        should_stop, patience, best_acc = should_terminate(patience, max_patience, local_test_acc, best_acc)

                        best_local_accs[net_id], best_local_accs_top5[net_id] = get_best_accs(local_test_acc, local_test_acc_top5, best_local_accs[net_id], best_local_accs_top5[net_id])
                        best_global_accs[net_id], best_global_accs_top5[net_id] = get_best_accs(global_test_acc, global_test_acc_top5, best_global_accs[net_id], best_global_accs_top5[net_id])

                        if local_test_acc > best_local_accs[net_id]:
                            best_local_accs[net_id] = local_test_acc
                        if global_test_acc > best_global_accs[net_id]:
                            best_global_accs[net_id] = global_test_acc

                        if should_stop:
                            print(f'Client {net_id} stopping after {i} epochs')
                            break

                            
    hparams = {k.replace('--', ''): v for k, v in vars(args).items()}
    hparams_str = str(hparams)
    
    file_exists = os.path.isfile(args.accuracies_file)
    has_header = False

    with open(args.accuracies_file, 'a', newline='') as file:
        writer = csv.writer(file)

        # Add header if file is new or doesn't have a header
        if not file_exists or os.stat(args.accuracies_file).st_size == 0:
            writer.writerow(['Client ID', 'Best Local Accuracy', 'Best Local Accuracy Top-5', 'Best Global Accuracy', 'Best Global Accuracy Top-5', 'Best Global Model Train', 'Best Global Model Test', 'Best Global Model Train', 'Best Global Model Test Top-5', 'Hyperparameters'])
            has_header = True

        # Write data rows
        for net_id in range(args.n_parties):
            writer.writerow([net_id, best_local_accs[net_id], best_local_accs_top5[net_id], best_global_accs[net_id], best_global_accs_top5[net_id], best_global_train, best_global_test, best_global_train_top5, best_global_test_top5, hparams_str])
        
def run_experiment(seed, alpha, dataset, args):
    args_copy = copy.deepcopy(args)
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    args_copy.seed = seed
    args_copy.alpha = alpha
    args_copy.dataset = dataset
    hyperparams = {k: v for k, v in vars(args_copy).items()}

    if not os.path.isfile(args.accuracies_file):
        with open(args.accuracies_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Client ID', 'Best Local Accuracy', 'Best Local Accuracy Top-5', 'Best Global Accuracy', 'Best Global Accuracy Top-5', 'Best Global Model Train', 'Best Global Model Test', 'Best Global Model Train', 'Best Global Model Test', 'Hyperparameters'])
            writer.writeheader()

    if not args.dont_skip:
        experiment_exists = False
        with open(args.accuracies_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:

                saved_hyperparams = eval(row['Hyperparameters'])
                exclude_keys = ['log_file_name', 'device']
                if all(saved_hyperparams.get(k) == v for k, v in hyperparams.items() if k not in exclude_keys):
                    print(f"Skipping experiment with hyperparameters: {hyperparams}")
                    return

    if not experiment_exists:
        print(f'Running experiment on dataset {args_copy.dataset} with seed {args_copy.seed} and dirich alpha {args_copy.alpha}')
        main(args_copy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication rounds')
    parser.add_argument('--sample_fraction', type=float, default=1.0, 
                        help='how many clients are sampled in each round')    
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.01], help='The parameters for the dirichlet distribution for data partitioning')
    
    parser.add_argument('--dataset', type=str, nargs='+', default=['cifar10'], help='The dataset to use')
    parser.add_argument('--datadir', type=str, required=False, default="../data/", help="Data directory")
    parser.add_argument('--partition', type=str, required=False, default='noniid', help='the data partitioning strategy')
    
    parser.add_argument('--alg', type=str, default='feduv',
                        help='federated learning framework: fedavg/fedprox/moon/freeze/feduv/sflv1/sflv2')    
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--split_layer', type=int, default=4, help='layer by which to split model in split learning')   
    
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')   
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for FedProx or MOON')    
    
    parser.add_argument('--std_coeff', type=float, default=2.5, help='the lambda parameter for FedUV')
    parser.add_argument('--unif_coeff', type=float, default=0.5, help='the mu parameter for FedUV')
    
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='input batch size for training')
    
    parser.add_argument('--load_first_net', type=int, default=1, 
                        help='whether load the first net as old net or not')
    parser.add_argument('--pool_option', type=str, default='FIFO', 
                        help='whether load the first net as old net or not')    
    
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')    
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    
    parser.add_argument('--simp_width', type=int, default=1, help='multiplier for CNN channel width (only for simple-cnn)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--reg', type=float, default=5e-4, help="L2 regularization strength")
    parser.add_argument('--temperature', type=float, default=0.5, 
                        help='the temperature parameter for contrastive loss')
    
    parser.add_argument('--logdir', type=str, required=False, default="./", help='Log directory path')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='The seed numbers')
    parser.add_argument('--device', type=str, default='cuda', help='The device to run the program (cuda/cpu)')

    parser.add_argument('--accuracies_file', type=str, default='logs/best_accuracies.csv', help='The file path to store the best accuracies')
    
    parser.add_argument('--dont_skip', action='store_true', help='Do not skip repetitious experiments')

    parser.add_argument('--n_train_workers', type=int, default=8, help='number of workers in a distributed cluster')   
    parser.add_argument('--n_test_workers', type=int, default=8, help='number of workers in a distributed cluster')   
    
    args = parser.parse_args()
    print(args.dataset)
    for dataset in args.dataset:
        print(dataset)
    
    for seed in args.seed:
        for alpha in args.alpha:
            for dataset in args.dataset:
                print(f'dataset: {dataset}')
                run_experiment(seed, alpha, dataset, args)

    
