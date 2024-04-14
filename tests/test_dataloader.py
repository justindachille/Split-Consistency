import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch

from utils.dataloader import record_net_data_stats, partition_data, get_dataloader, FedAvg
from utils.dataloader import load_cifar10_data, load_stl10_data, load_cifar100_data


class TestDataLoading(unittest.TestCase):

    def test_record_net_data_stats(self):
        y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        net_dataidx_map = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        logdir = './'
        logger = Mock()

        net_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir, logger)

        self.assertEqual(len(net_cls_counts), 3)
        self.assertDictEqual(net_cls_counts[0], {0: 1, 1: 1, 2: 1})
        self.assertDictEqual(net_cls_counts[1], {0: 1, 1: 1, 2: 1})
        self.assertDictEqual(net_cls_counts[2], {0: 1, 1: 1, 2: 1})

    @patch('utils.dataloader.CIFAR10')
    def test_load_cifar10_data(self, mock_cifar10):
        args = Mock()
        datadir = './'
        X_train, y_train, X_test, y_test = load_cifar10_data(args, datadir)

        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_test)

    def test_partition_data_homo(self):
        n_parties = 2
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[9, 10], [11, 12]])
        y_test = np.array([0, 1])
        partition = 'homo'

        X_train, y_train, X_test, y_test, net_dataidx_map, _ = partition_data(
            None, 'cifar10', '../data/', './', partition, n_parties, beta=0.4
        )
        self.assertEqual(len(net_dataidx_map), n_parties)
        self.assertEqual(sum(len(idx) for idx in net_dataidx_map.values()), len(y_train))
        print('data equal')
        
    def test_get_dataloader(self):
        ds_name = 'cifar10'
        datadir = './'
        train_bs = 32
        test_bs = 64
        X_train = np.random.rand(100, 32, 32, 3)
        y_train = np.random.randint(0, 10, 100)
        X_test = np.random.rand(20, 32, 32, 3)
        y_test = np.random.randint(0, 10, 20)

        train_dl, test_dl, train_ds, test_ds, test_dl_local = get_dataloader(
            ds_name, datadir, train_bs, test_bs, X_train, y_train, X_test, y_test
        )

        self.assertIsInstance(train_dl, torch.utils.data.DataLoader)
        self.assertIsInstance(test_dl, torch.utils.data.DataLoader)
        self.assertIsInstance(train_ds, torch.utils.data.Dataset)
        self.assertIsInstance(test_ds, torch.utils.data.Dataset)
        self.assertIsNone(test_dl_local)

    def test_FedAvg(self):
        w = [
            {'layer1.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
            {'layer1.weight': torch.tensor([[5.0, 6.0], [7.0, 8.0]])}
        ]

        w_avg = FedAvg(w)

        expected_w_avg = {'layer1.weight': torch.tensor([[3.0, 4.0], [5.0, 6.0]])}
        self.assertTrue(torch.allclose(w_avg['layer1.weight'], expected_w_avg['layer1.weight']))

    def test_augmentation_cifar10(self):
        ds_name = 'cifar10'
        datadir = '../data/'
        train_bs = 32
        test_bs = 64
        args = Mock()
        args.datadir = datadir

        X_train, y_train, X_test, y_test = load_cifar10_data(args, datadir)
        train_dl, test_dl, train_ds, test_ds, test_dl_local = get_dataloader(
            ds_name, datadir, train_bs, test_bs, X_train, y_train, X_test, y_test
        )

        
        with self.subTest(split='train'):
            self._check_augmentation(train_ds, active=True)
        with self.subTest(split='test'):
            self._check_augmentation(test_ds, active=False)

    def test_augmentation_cifar100(self):
        ds_name = 'cifar100'
        datadir = '../data/'
        train_bs = 32
        test_bs = 64
        args = Mock()
        args.datadir = datadir

        X_train, y_train, X_test, y_test = load_cifar100_data(args, datadir)
        train_dl, test_dl, train_ds, test_ds, test_dl_local = get_dataloader(
            ds_name, datadir, train_bs, test_bs, X_train, y_train, X_test, y_test
        )
        
        with self.subTest(split='train'):
            self._check_augmentation(train_ds, active=True)
        with self.subTest(split='test'):
            self._check_augmentation(test_ds, active=False)

    def test_augmentation_stl10(self):
        ds_name = 'stl10'
        datadir = '../data/'
        train_bs = 32
        test_bs = 64
        args = Mock()
        args.datadir = datadir

        X_train, y_train, X_test, y_test = load_stl10_data(args, datadir)
        train_dl, test_dl, train_ds, test_ds, test_dl_local = get_dataloader(
            ds_name, datadir, train_bs, test_bs, X_train, y_train, X_test, y_test
        )

        with self.subTest(split='train'):
            self._check_augmentation(train_ds, active=True)
        with self.subTest(split='test'):
            self._check_augmentation(test_ds, active=False)

    def _check_augmentation(self, data, active):
        are_same = []
        for i in range(min(len(data), 100)):  # Check a subset of the dataset
            sample_1, _ = data[i]
            sample_2, _ = data[i]
            are_same.append(torch.allclose(sample_1, sample_2, atol=1e-6))

        if active:
            self.assertFalse(all(are_same))
        else:
            self.assertTrue(all(are_same))
if __name__ == '__main__':
    unittest.main()