#===========================================================
# Federated learning: ResNet18 on HAM10000
# HAM10000 dataset: Tschandl, P.: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation 
# ===========================================================
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob 
import math
import random
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))    


#===================================================================  
program = "FL ResNet18 on HAM10000"
print(f"---------{program}----------")              # this is to identify the program in the slurm outputs files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# To print in color during test/train 
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    


#===================================================================
# No. of users
num_users = 10
epochs = 200
frac = 1
lr = 0.01

#==============================================================================================================
#                                  Client Side Program 
#==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Client-side functions associated with Training and Testing
class LocalUpdate(object):
    def __init__(self, idx, lr, device, dataset_train = None, dataset_test = None, idxs = None, idxs_test = None):
        self.idx = idx
        self.device = device
        self.lr = lr
        self.local_ep = 5
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size = 128, shuffle = True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = 100, shuffle = True)

    def train(self, net):
        net.train()
        # train and update
        #optimizer = torch.optim.SGD(net.parameters(), lr = self.lr, momentum = 0.5)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        epoch_acc = []
        epoch_loss = []
        for iter in range(self.local_ep):
            batch_acc = []
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                #---------forward prop-------------
                fx = net(images)
                
                # calculate loss
                loss = self.loss_func(fx, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx, labels)
                
                #--------backward prop--------------
                loss.backward()
                optimizer.step()
                              
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            
            
            prRed('Client{} Train => Local Epoch: {}  \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx,
                        iter, acc.item(), loss.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
    
    def evaluate(self, net):
        net.eval()
           
        epoch_acc = []
        epoch_loss = []
        with torch.no_grad():
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                #---------forward prop-------------
                fx = net(images)
                
                # calculate loss
                loss = self.loss_func(fx, labels)
                # calculate accuracy
                acc = calculate_accuracy(fx, labels)
                                 
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
            
            prGreen('Client{} Test =>                     \tLoss: {:.4f} \tAcc: {:.3f}'.format(self.idx, loss.item(), acc.item()))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_acc.append(sum(batch_acc)/len(batch_acc))
        return sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)


#=============================================================================
#                         Data loading 
#============================================================================= 



#==============================================================
# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform = None):
        
        self.df = df
        self.transform = transform
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y

#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID HAM10000 datasets will be created based on this

def cifar_user_dataset(dataset, num_users, noniid_fraction):
    """
    Sample a 'fraction' of non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param fraction:
    :return:
    """

    # initialization
    total_items = len(dataset)
    num_noniid_items = len(dataset) * noniid_fraction
    num_iid_items = total_items - num_noniid_items
    dict_users = list()
    for ii in range(num_users):
        dict_users.append(list())
    idxs = [i for i in range(len(dataset))]

    # IID
    if num_iid_items != 0:
        per_user_iid_items = int(num_iid_items / num_users)
        for ii in range(num_users):
            tmp_set = set(np.random.choice(idxs, per_user_iid_items, replace=False))
            dict_users[ii] += tmp_set
            idxs = list(set(idxs) - tmp_set)

    # NON-IID
    if num_noniid_items != 0:

        num_shards = num_users  # each user has one shard
        per_shards_num_imgs = int(num_noniid_items / num_shards)
        idx_shard = [i for i in range(num_shards)]
        labels = list()
        for ii in range(len(idxs)):
            labels.append(dataset[idxs[ii]][1])
        print(labels)
        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        # for i in range(len(idxs_labels)):
        #     print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
        #     print(idxs_labels[i])
        idxs = idxs_labels[0, :]

        # divide and assign
        i = 0
        while idx_shard:
            print(idx_shard)
            rand_idx = np.random.choice(idx_shard, 1, replace=False)
            rand_idx[0] = idx_shard[0]
            # rand_idx.append(idx_shard[0])
            print(rand_idx)
            idx_shard = list(set(idx_shard) - set(rand_idx))
            dict_users[i].extend(idxs[int(rand_idx) * per_shards_num_imgs: (int(rand_idx) + 1) * per_shards_num_imgs])
            i = divmod(i + 1, num_users)[1]

    '''
    for ii in range(num_users):
        tmp = list()
        for jj in range(len(dict_users[ii])):
            tmp.append(dataset[dict_users[ii][jj]][1])
        tmp.sort()
        print(tmp)
    '''
    return dict_users

def dataset_iid(dataset, num_users):
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
dataset_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
dataset_test = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

# -----------------------------------------------
# with open('beta=0.1.pkl', 'rb') as file:
#     dict_users=pickle.load(file)
# dict_users=cifar_user_dataset(dataset_train,num_users,0)
with open('cifar0.1.txt', 'r') as file:
    content = file.read()
dict_users = eval(content)
dict_users_test = dataset_iid(dataset_test, num_users)


#====================================================================================================
#                               Server Side Program
#====================================================================================================
def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

#=============================================================================
#                    Model definition: ResNet18
#============================================================================= 
# building a ResNet18 Architecture


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



net_glob=ResNet18(ResidualBlock)
if torch.cuda.device_count() > 1:
    print("We use",torch.cuda.device_count(), "GPUs")
    net_glob = nn.DataParallel(net_glob)   # to use the multiple GPUs 

net_glob.to(device)
print(net_glob)      

#===========================================================================================
# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
#====================================================
net_glob.train()
# copy weights
w_glob = net_glob.state_dict()

loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []

for iter in range(epochs):
    w_locals, loss_locals_train, acc_locals_train, loss_locals_test, acc_locals_test = [], [], [], [], []
    m = max(int(frac * num_users), 1)
    idxs_users = np.random.choice(range(num_users), m, replace = False)
    
    # Training/Testing simulation
    for idx in idxs_users: # each client
        local = LocalUpdate(idx, lr, device, dataset_train = dataset_train, dataset_test = dataset_test, idxs = dict_users[idx], idxs_test = dict_users_test[idx])
        # Training ------------------
        w, loss_train, acc_train = local.train(net = copy.deepcopy(net_glob).to(device))
        w_locals.append(copy.deepcopy(w))
        loss_locals_train.append(copy.deepcopy(loss_train))
        acc_locals_train.append(copy.deepcopy(acc_train))
        # Testing -------------------
        loss_test, acc_test = local.evaluate(net = copy.deepcopy(net_glob).to(device))
        loss_locals_test.append(copy.deepcopy(loss_test))
        acc_locals_test.append(copy.deepcopy(acc_test))
        
        
    
    # Federation process
    w_glob = FedAvg(w_locals)
    print("------------------------------------------------")
    print("------ Federation process at Server-Side -------")
    print("------------------------------------------------")
    
    # update global model --- copy weight to net_glob -- distributed the model to all users
    net_glob.load_state_dict(w_glob)
    
    # Train/Test accuracy
    acc_avg_train = sum(acc_locals_train) / len(acc_locals_train)
    acc_train_collect.append(acc_avg_train)
    acc_avg_test = sum(acc_locals_test) / len(acc_locals_test)
    acc_test_collect.append(acc_avg_test)
    
    # Train/Test loss
    loss_avg_train = sum(loss_locals_train) / len(loss_locals_train)
    loss_train_collect.append(loss_avg_train)
    loss_avg_test = sum(loss_locals_test) / len(loss_locals_test)
    loss_test_collect.append(loss_avg_test)
    
    
    print('------------------- SERVER ----------------------------------------------')
    print('Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_train, loss_avg_train))
    print('Test:  Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_test, loss_avg_test))
    print('-------------------------------------------------------------------------')
   

#===================================================================================     

print("Training and Evaluation completed!")    

#===============================================================================
# Save output data to .excel file (we use for comparision plots)
# round_process = [i for i in range(1, len(acc_train_collect)+1)]
# df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_test':acc_test_collect})
# file_name = program+".xlsx"
# df.to_excel(file_name, sheet_name= "v1_test", index = False)
print(loss_train_collect)
print(loss_test_collect)
print(acc_train_collect)
print(acc_test_collect)

#=============================================================================
#                         Program Completed
#============================================================================= 





