import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import torch
from nets.models import SimpleCNN, ResNet_18, ResNet_50, AlexNetClient, AlexNetServer, AlexNet, ResNet_18_client_side, ResNet_18_server_side, ResNet_50_client_side, ResNet_50_server_side, VGG16, VGG16_client_side, VGG16_server_side, ResidualBlock

class TestModels(unittest.TestCase):
    
    # Dummy test images
    NUM_TEST_IMAGES = 4
    CIFAR10_IMAGE = torch.randn(NUM_TEST_IMAGES, 3, 32, 32)
    IMAGENET_IMAGE = torch.randn(NUM_TEST_IMAGES, 3, 224, 224)
    
    # Number of classes
    NUM_CLASSES_CIFAR10 = 10
    NUM_CLASSES_IMAGENET = 1000
    

    def test_SimpleCNN_shape(self):
        net = SimpleCNN(output_dim=64, n_classes=self.NUM_CLASSES_CIFAR10, width=1)
        inputs = self.CIFAR10_IMAGE
        _, _, outputs, _, _ = net(inputs)
        self.assertEqual(outputs.shape, (self.NUM_TEST_IMAGES, self.NUM_CLASSES_CIFAR10))

    def test_ResNet_18_shape(self):
        args = type('', (), {})()
        args.out_dim = 64
        net = ResNet_18(args, num_class=2)
        inputs = self.CIFAR10_IMAGE
        _, _, outputs, _, _ = net(inputs)
        self.assertEqual(outputs.shape, (self.NUM_TEST_IMAGES, 2))

    def test_ResNet_50_shape(self):
        args = type('', (), {})()
        args.out_dim = 64
        net = ResNet_50(args, num_class=2)
        inputs = self.CIFAR10_IMAGE
        _, _, outputs, _, _ = net(inputs)
        self.assertEqual(outputs.shape, (self.NUM_TEST_IMAGES, 2))

    def test_AlexNet_shape(self):
        args = type('', (), {})()
        net = AlexNet(args, num_classes=self.NUM_CLASSES_CIFAR10)
        inputs = self.CIFAR10_IMAGE
        _, _, outputs, _, _ = net(inputs)
        self.assertEqual(outputs.shape, (self.NUM_TEST_IMAGES, self.NUM_CLASSES_CIFAR10))

    def test_VGG16_shape(self):
        net = VGG16(n_classes=self.NUM_CLASSES_IMAGENET)
        inputs = self.IMAGENET_IMAGE
        _, _, outputs, _, _ = net(inputs)
        self.assertEqual(outputs.shape, (self.NUM_TEST_IMAGES, self.NUM_CLASSES_IMAGENET))

    def test_ResNet_18_split_shape(self):
        for split_layer in range(1, 5):
            with self.subTest(split_layer=split_layer):
#                 print(f'testing split layer: {split_layer}')
                args = type('', (), {})()
                args.split_layer = split_layer
                client_net = ResNet_18_client_side(ResidualBlock, args)
                server_net = ResNet_18_server_side(ResidualBlock, args, num_classes=100)

                inputs = self.CIFAR10_IMAGE
                client_outputs = client_net(inputs)
                server_outputs = server_net(client_outputs)

                self.assertEqual(server_outputs.shape, (self.NUM_TEST_IMAGES, 100))

                # Check if the output shape of client_net matches the expected input shape of the server model
                expected_inchannel = 64 * (2 ** (split_layer - 1))
                self.assertEqual(client_outputs.shape[1], expected_inchannel)

    def test_ResNet_50_split_shape(self):
        for split_layer in range(1, 5):
            with self.subTest(split_layer=split_layer):
#                 print(f'testing split layer: {split_layer}')
                args = type('', (), {})()
                args.split_layer = split_layer
                args.out_dim = 64
                client_net = ResNet_50_client_side(args)
                server_net = ResNet_50_server_side(args, num_classes=2)

                inputs = self.CIFAR10_IMAGE
                client_outputs = client_net(inputs)
                server_outputs = server_net(client_outputs)

                self.assertEqual(server_outputs.shape, (self.NUM_TEST_IMAGES, 2))

                # Check if the output shape of client_net matches the input shape of the first layer in server_net
                if split_layer == 4:
                    self.assertEqual(client_outputs.shape[1], 2048)
                else:
                    server_first_layer = server_net.layers[0][0].conv1
                    self.assertEqual(client_outputs.shape[1], server_first_layer.in_channels)

    def test_VGG16_split_shape(self):
        for split_layer in range(1, 6):
            with self.subTest(split_layer=split_layer):
#                 print(f'testing split layer: {split_layer}')
                args = type('', (), {})()
                args.split_layer = split_layer
                client_net = VGG16_client_side(args)
                server_net = VGG16_server_side(args, num_classes=self.NUM_CLASSES_IMAGENET)
                inputs = self.IMAGENET_IMAGE
                client_outputs = client_net(inputs)
                server_outputs = server_net(client_outputs)
                self.assertEqual(server_outputs.shape, (self.NUM_TEST_IMAGES, self.NUM_CLASSES_IMAGENET))

                
if __name__ == '__main__':
    unittest.main()