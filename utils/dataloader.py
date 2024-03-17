import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder, DatasetFolder, utils
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from glob import glob
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
from collections import Counter

def record_net_data_stats(y_train, net_dataidx_map, logdir, logger):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def load_stl10_data(args, datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    stl10_train_ds = STL10(args.datadir, split='train', download=True, transform=transform)
    stl10_test_ds = STL10(args.datadir, split='test', download=True, transform=transform)

    X_train, y_train = stl10_train_ds.data, stl10_train_ds.labels
    X_test, y_test = stl10_test_ds.data, stl10_test_ds.labels

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.transpose(X_train, (0,2,3,1))
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.transpose(X_test, (0,2,3,1))

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(args, datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(args, datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.targets
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.targets

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(args, datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/val/images', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def load_feature_shift(args):

    if args.dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch'] # pacs
        
    elif args.dataset == 'ham':
        domains = ['vidir_molemax', 'vidir_modern', 'rosendahl', 'vienna_dias'] # HAM
        
    elif args.dataset == 'office':
        domains = ['Art', 'Clipart', 'Product', 'Real'] # office

    all_train_ds = []
    for domain in domains:
        all_train_ds.append(dl_obj(f"{args.datadir}/{args.dataset}/train/{domain}", transform=transform_train))

    train_ds_global = torch.utils.data.ConcatDataset(all_train_ds)

    train_dl_global = data.DataLoader(dataset=train_ds_global, batch_size=args.batch_size, num_workers=8, shuffle=True, 
                             pin_memory=True, persistent_workers=True
                             )

    test_ds_global = dl_obj(f"{args.datadir}/{args.dataset}/test", transform=transform_test)
    test_dl = data.DataLoader(dataset=test_ds_global, batch_size=args.batch_size, num_workers=4, shuffle=False, 
                             pin_memory=True, persistent_workers=True
                             )
    
    return all_train_ds, train_ds_global, train_dl_global, test_ds_global, test_dl


def partition_data(args, dataset, datadir, logdir, partition, n_parties, beta=0.4, logger=None):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(args, datadir)
    elif dataset == 'stl10':
        X_train, y_train, X_test, y_test = load_stl10_data(args, datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(args, datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(args, datadir)
    elif dataset == 'ham10000':
        return Ham10000().preprocess_data()

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir" or partition == "noniid" or partition == 'subsample':
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            
            if partition == 'subsample':
                assert(args.n_parties == 2)
                idx_batch[0] = np.random.choice(idx_batch[0], size=5000, replace=False)
                idx_batch[1] = np.random.choice(idx_batch[1], size=8000, replace=False)


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            

    ########### SEE IF WE NEED THIS LATER ##############
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir, logger)
    
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

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

def dataset_non_iid_both(dataset_train, dataset_test, num_users, beta=0.5, subsample=[5000, 8000]):
    dict_users_train, dirichlet_dist = dataset_non_iid(dataset_train, num_users, beta=beta, subsample_sizes=subsample)
    dict_users_test, _ = dataset_non_iid(dataset_test, num_users, beta=beta, dirichlet_dist_to_use=dirichlet_dist, subsample_sizes=subsample)
    return dict_users_train, dict_users_test

def dataset_non_iid(dataset, num_users, beta=0.5, dirichlet_dist_to_use=None, subsample_sizes=[5000, 8000]):
    num_classes = len(set([label for _, label in dataset]))
    dict_users = {}

    if dirichlet_dist_to_use is None:
        dirichlet_dist = np.random.dirichlet([beta] * num_users, num_classes)
        dirichlet_dist[np.isinf(dirichlet_dist)] = 1.0
        dirichlet_dist = np.nan_to_num(dirichlet_dist, nan=0.0)
        dirichlet_dist /= dirichlet_dist.sum(axis=1, keepdims=True)
    else:
        dirichlet_dist = dirichlet_dist_to_use

    class_indices = [[] for _ in range(num_classes)]

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    for j in range(num_users):
        dict_users[j] = []

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        num_total = len(class_indices[c])
        
        end_idx = 0
        for j in range(num_users):
            start_idx = end_idx
            end_idx += int(np.ceil(dirichlet_dist[c, j] * num_total))
            indices_to_add = class_indices[c][start_idx:end_idx]
            dict_users[j].extend(indices_to_add)

    # Now, randomly subsample each client's data while maintaining the distribution
    for j in range(num_users):
        np.random.shuffle(dict_users[j])
        subsample_size = subsample_sizes[j % len(subsample_sizes)]
        dict_users[j] = set(dict_users[j][:subsample_size])

    return dict_users, dirichlet_dist

class Ham10000:
    def __init__(self):
        self.df, self.df_test = self.load_data()
        self.print_data_info()
        self.train = self.df
        self.dataset_train, self.dataset_test, _, _, _ = self.preprocess_data()   

    def load_data(self):
        df = pd.read_csv('data/HAM10000_metadata.csv')
        df_test = pd.read_csv('data/ISIC2018_Task3_Test_GroundTruth.csv')

        lesion_type = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }

        def add_images(df):
            imageid_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join("data", '*', '*.jpg'))}
            df['path'] = df['image_id'].map(imageid_path.get)
            df['cell_type'] = df['dx'].map(lesion_type.get)
            df['target'] = pd.Categorical(df['cell_type']).codes
            return df
        
        df = add_images(df)
        df_test = add_images(df_test)

        return df, df_test

    def print_data_info(self):
        print(self.df['cell_type'].value_counts())
        print(self.df['target'].value_counts())

    def preprocess_data(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                               transforms.RandomVerticalFlip(),
                                               transforms.Pad(3),
                                               transforms.RandomRotation(10),
                                               transforms.CenterCrop(64),
                                               transforms.ToTensor(), 
                                               transforms.Normalize(mean=mean, std=std)])

        test_transforms = transforms.Compose([
                            transforms.Pad(3),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)])
        
        dataset_train = SkinData(self.train, transform=train_transforms)
        dataset_test = SkinData(self.df_test, transform=test_transforms)

        def create_client_dataloaders(dataset, batch_size=32):
            # Determine the number of samples for each client
            num_samples_client1 = np.random.choice([4000, 6000])
            num_samples_client2 = np.random.choice([4000, 6000])

            # Generate random indices for sampling
            indices = torch.randperm(len(dataset)).tolist()

            # Split indices for each client
            client1_indices = indices[:num_samples_client1]
            client2_indices = indices[num_samples_client1:num_samples_client1 + num_samples_client2]

            # Create subsets for each client
            client1_dataset = Subset(dataset, client1_indices)
            client2_dataset = Subset(dataset, client2_indices)

            # Create dataloaders for each client
            client1_loader = DataLoader(client1_dataset, batch_size=batch_size, shuffle=True)
            client2_loader = DataLoader(client2_dataset, batch_size=batch_size, shuffle=True)

            return client1_loader, client1_indices, client2_loader, client2_indices
        client1_loader, client1_indices, client2_loader, client2_indices = create_client_dataloaders(dataset_train)

        net_dataidx_map = {
            0: client1_indices,
            1: client2_indices
        }

        return dataset_train, dataset_test, client1_loader, client2_loader, net_dataidx_map


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
        
        
class CustomDataset(Dataset):
    def __init__(self, data, targets, transforms=None):
        
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transforms = transforms

        print(self.data.shape)
        print(self.targets.shape)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        if self.transforms is not None:
            batch_x, batch_y = self.transforms(self.data[index]), self.targets[index]
        else:
            batch_x, batch_y = self.data[index], self.targets[index]

        return batch_x, batch_y
    

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        
        self.idxs = None
        if idxs is not None:
            self.idxs = list(idxs)
            
    def __len__(self):
        if self.idxs is not None:
            return len(self.idxs)
        else:
            return len(self.dataset)

    def __getitem__(self, item):
        if self.idxs is not None:
            image, label = self.dataset[self.idxs[item]]
        else:
            image, label = self.dataset[item]
        
        return image, label
    
    
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)     

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def get_label_proportions(y_data, dataidxs):
    labels = np.array(y_data)[dataidxs]
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    label_proportions = [f"{label}:{count}" for label, count in zip(unique_labels, counts)]
    label_proportions_str = ", ".join(label_proportions)
    print(f'Label quantities: {label_proportions_str}')

    label_prop_percentages = [count / total_samples for count in counts]
    return unique_labels, counts, label_prop_percentages

def get_stratified_test_split(y_test, unique_labels, label_prop_percentages):
    test_stratified_idxs = []
    unique_labels_test, counts_test = np.unique(y_test, return_counts=True)
    label_prop_dict = dict(zip(unique_labels, label_prop_percentages))
    for label, count in zip(unique_labels_test, counts_test):
        if label in label_prop_dict:
            prop = label_prop_dict[label]
            label_idxs = np.where(y_test == label)[0]
            label_test_count = int(prop * len(label_idxs))
            test_stratified_idxs.extend(np.random.choice(label_idxs, size=label_test_count, replace=False))
    return test_stratified_idxs

def get_dataloader(ds_name, datadir, train_bs, test_bs, X_train=None, y_train=None, X_test=None, y_test=None, dataidxs=None, noise_level=0, partition=False):

    test_dl_local = None
    
    if ds_name in ('cifar10', 'cifar100'):
        if ds_name == 'cifar10':
            
            dataset=CIFAR10
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2023, 0.1994, 0.2010])
#             transform_train = transforms.Compose([
#                 transforms.RandomCrop(32, padding=4),
#                 #transforms.ColorJitter(brightness=noise_level),
#                 transforms.RandomHorizontalFlip(),
#                 #transforms.RandomRotation(15),
#                 transforms.ToTensor(),
#                 normalize
#             ])
            transform_train = transforms.Compose([
                transforms.ToTensor(),  # Convert to tensor first
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        elif ds_name == 'cifar100':

            dataset=CIFAR100
            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            transform_train = transforms.Compose([
               transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            
        
        train_ds = CustomDataset(X_train, y_train, transforms=transform_train)
        test_ds  = CustomDataset(X_test, y_test, transforms=transform_test)

        
        train_ds = DatasetSplit(train_ds, dataidxs)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, num_workers=8, drop_last=True, shuffle=True, pin_memory=True, persistent_workers =True, worker_init_fn=set_worker_sharing_strategy)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, num_workers=8, shuffle=False,persistent_workers=True, worker_init_fn=set_worker_sharing_strategy)
        
        if partition == 'subsample' and dataidxs is not None:
            unique_labels, counts, label_prop_percentages = get_label_proportions(y_train, dataidxs)
            test_stratified_idxs = get_stratified_test_split(y_test, unique_labels, label_prop_percentages)
            test_ds_local = DatasetSplit(test_ds, test_stratified_idxs)
            test_dl_local = data.DataLoader(dataset=test_ds_local, batch_size=test_bs, num_workers=8, shuffle=False, persistent_workers=True, worker_init_fn=set_worker_sharing_strategy)

            # Print test set indices/proportions
            unique_labels_test, counts_test = np.unique(np.array(y_test)[test_stratified_idxs], return_counts=True)
            test_label_proportions = [f"{label}:{count}" for label, count in zip(unique_labels_test, counts_test)]
            test_label_proportions_str = ", ".join(test_label_proportions)
            print(f'Test set label quantities: {test_label_proportions_str}')        

    elif ds_name == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),              
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_ds = dl_obj(datadir+'tiny-imagenet-200/train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'tiny-imagenet-200/val_again/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, num_workers=8, drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True, worker_init_fn=set_worker_sharing_strategy)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, num_workers=8, shuffle=False, pin_memory=True, persistent_workers=True, worker_init_fn=set_worker_sharing_strategy)
        
        
    elif ds_name == 'imbd':
        train_ds = CustomDataset(X_train, y_train)
        test_ds  = CustomDataset(X_test, y_test)

        train_ds = DatasetSplit(train_ds, dataidxs)
        #test_ds  = DatasetSplit(test_ds,  dataidxs)

        train_dl = torch.utils.data.DataLoader(dataset=train_ds, 
                                   batch_size=train_bs, 
                                   num_workers=8, 
                                   drop_last=True, 
                                   shuffle=True, 
                                   pin_memory=True, 
                                   persistent_workers =True, worker_init_fn=set_worker_sharing_strategy)

        test_dl = torch.utils.data.DataLoader(dataset=test_ds, 
                                   batch_size=test_bs, 
                                   num_workers=8, 
                                   drop_last=True, 
                                   shuffle=False, worker_init_fn=set_worker_sharing_strategy)        

        
    elif ds_name == 'stl10':
        
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])        
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        train_ds = CustomDataset(X_train, y_train, transforms=transform_train)
        test_ds  = CustomDataset(X_test, y_test, transforms=transform_test)

        train_ds = DatasetSplit(train_ds, dataidxs)
                

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, num_workers=8, drop_last=True, shuffle=True, pin_memory=True, persistent_workers =True, worker_init_fn=set_worker_sharing_strategy)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, num_workers=8, shuffle=False,persistent_workers=True, worker_init_fn=set_worker_sharing_strategy)
        

    return train_dl, test_dl, train_ds, test_ds, test_dl_local

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
