from nets.resnet_utils import BasicBlock, Bottleneck, conv3x3, conv1x1
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, output_dim=64, n_classes=10, width=1):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16*width, kernel_size=3, bias=False) 
        self.conv1_bn = nn.BatchNorm2d(16*width)
        self.conv2 = nn.Conv2d(16*width, 16*width, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(16*width)        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(16*width, 32*width, 3, stride=2, bias=False) 
        self.conv3_bn = nn.BatchNorm2d(32*width)
        
        self.conv4 = nn.Conv2d(32*width, 32*width, 3, bias=False)
        self.conv4_bn = nn.BatchNorm2d(32*width)
        self.conv5 = nn.Conv2d(32*width, 32*width, 1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(32*width)
        
        self.conv6 = nn.Conv2d(32*width, 64*width, 3, stride=2, bias=False) 
        self.conv6_bn = nn.BatchNorm2d(64*width)        

        self.conv7 = nn.Conv2d(64*width, 64*width, 3, bias=False)
        self.conv7_bn = nn.BatchNorm2d(64*width)
        
        self.conv8 = nn.Conv2d(64*width, 64*width, 1, bias=False)
        self.conv8_bn = nn.BatchNorm2d(64*width)        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(64*width, output_dim)
        self.fc1_bn = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.fc2_bn = nn.BatchNorm1d(output_dim)
        self.fc3 = nn.Linear(output_dim, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = self.relu(x)
        
        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = self.relu(x)
        
        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = self.relu(x)        

        x = self.avgpool(x)

        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.relu(x)        

        y = self.fc3(x)        

        return 0, x, y, 0, 0
    

class ResNet_18(nn.Module):
    def __init__(self, args,  num_class=2):
        super(ResNet_18, self).__init__()
        
        
        norm_layer = nn.BatchNorm2d
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            
        )
        #self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        
        downsample_l1 = nn.Sequential(conv1x1(in_planes=64, out_planes=64, stride=2), norm_layer(64))
        self.layer1 = nn.Sequential(
            BasicBlock(inplanes=64, planes=64, stride=2, downsample=downsample_l1, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),      
            
            BasicBlock(inplanes=64, planes=64, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
               

        )      
        
        downsample_l2 = nn.Sequential(conv1x1(in_planes=64, out_planes=128, stride=2), norm_layer(128))
        self.layer2 = nn.Sequential(
            
            BasicBlock(inplanes=64, planes=128, stride=2, downsample=downsample_l2, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            BasicBlock(inplanes=128, planes=128, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer)        

        )            

        
        downsample_l3 = nn.Sequential(conv1x1(in_planes=128, out_planes=256, stride=2), norm_layer(256))
        self.layer3 = nn.Sequential(
            
            BasicBlock(inplanes=128, planes=256, stride=2, downsample=downsample_l3, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            BasicBlock(inplanes=256, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                          
        )                 

        
        downsample_l4 = nn.Sequential(conv1x1(in_planes=256, out_planes=512, stride=2), norm_layer(512))
        self.layer4 = nn.Sequential(
            
            BasicBlock(inplanes=256, planes=512, stride=2, downsample=downsample_l4, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            BasicBlock(inplanes=512, planes=512, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            

        )         
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        num_ftrs = args.out_dim
        self.fc1 = nn.Linear(512, num_ftrs)
        self.fc1_bn = nn.BatchNorm1d(num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2_bn = nn.BatchNorm1d(num_ftrs)
        self.fc3 = nn.Linear(num_ftrs, num_class)


        self.relu = nn.ReLU()
        
    def forward(self, x):
        out1 = self.initial_conv(x)
        #x = self.max_pool(out1)
        
        out2 = self.layer1(out1)
        x = self.layer2(out2)
        x = self.layer3(x)
        x = self.layer4(x)
      
        x = self.avgpool(x)
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        preds = self.fc3(x)
        
        return x, x, preds, out1, out2
    
    
class ResNet_50(nn.Module):
    def __init__(self, args,  num_class=2):
        super(ResNet_50, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        #self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2, padding=1)
        
        downsample_l1 = nn.Sequential(conv1x1(in_planes=64, out_planes=256, stride=1), norm_layer(256))
        self.layer1 = nn.Sequential(
            
            Bottleneck(inplanes=64, planes=64, stride=1, downsample=downsample_l1, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            Bottleneck(inplanes=256, planes=64, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
            Bottleneck(inplanes=256, planes=64, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                        

        )      
        
        downsample_l2 = nn.Sequential(conv1x1(in_planes=256, out_planes=512, stride=2), norm_layer(512))
        self.layer2 = nn.Sequential(
            
            Bottleneck(inplanes=256, planes=128, stride=2, downsample=downsample_l2, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
            Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),          

        )            

        
        downsample_l3 = nn.Sequential(conv1x1(in_planes=512, out_planes=1024, stride=2), norm_layer(1024))
        self.layer3 = nn.Sequential(
            
            Bottleneck(inplanes=512, planes=256, stride=2, downsample=downsample_l3, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),   
            Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                

        )                 

        
        downsample_l4 = nn.Sequential(conv1x1(in_planes=1024, out_planes=2048, stride=2), norm_layer(2048))
        self.layer4 = nn.Sequential(
            
            Bottleneck(inplanes=1024, planes=512, stride=2, downsample=downsample_l4, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            
            Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
            Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, 
                       groups=1, base_width=64, dilation=1, norm_layer=norm_layer),

        )             
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        ### Projection head
        num_ftrs = args.out_dim
        self.fc1 = nn.Linear(2048, num_ftrs)
        self.fc1_bn = nn.BatchNorm1d(num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2_bn = nn.BatchNorm1d(num_ftrs)
        self.fc3 = nn.Linear(num_ftrs, num_class)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        out1 = self.initial_conv(x)
        #x = self.max_pool(out1)
        
        out2 = self.layer1(out1)
        x = self.layer2(out2)
        x = self.layer3(x)
        x = self.layer4(x)

                
        x = self.avgpool(x)
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        preds = self.fc3(x)
        
        return x, x, preds, out1, out2
    
class AlexNetClient(nn.Module, ):
    def __init__(self, args, num_classes):
        super(AlexNetClient, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x
    
class AlexNetServer(nn.Module):
    def __init__(self, args, num_classes=10):
        super(AlexNetServer, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super(AlexNet, self).__init__()
        # Feature extraction part from AlexNetClient
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)  # Flatten the output for the classifier
        x = self.classifier(x)
        return 0, 0, x, 0, 0
  

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

class ResNet_18_client_side(nn.Module):
    def __init__(self, ResidualBlock, args):
        super(ResNet_18_client_side, self).__init__()
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
#         self.fc = nn.Linear(512, num_classes)

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
        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        #out = self.fc(out)
        return out    
    
# class ResNet_18_client_side(nn.Module):
#     def __init__(self, ResidualBlock, args):
#         super(ResNet_18_client_side, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layers = nn.ModuleList()
#         if args.split_layer >= 1:
#             self.layers.append(self.make_layer(ResidualBlock, 64, 2, stride=1))
#         if args.split_layer >= 2:
#             self.layers.append(self.make_layer(ResidualBlock, 128, 2, stride=2))
#         if args.split_layer >= 3:
#             self.layers.append(self.make_layer(ResidualBlock, 256, 2, stride=2))
#         if args.split_layer >= 4:
#             self.layers.append(self.make_layer(ResidualBlock, 512, 2, stride=2))

#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         for layer in self.layers:
#             out = layer(out)
#         return out

class ResNet_18_server_side(nn.Module):
    def __init__(self, ResidualBlock, args, num_classes=100):
        super(ResNet_18_server_side, self).__init__()
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
        #out = self.conv1(x)
        #out = self.layer1(out)
        # out = self.layer2(x)
        # out = self.layer3(x)
        # out = self.layer4(out)
        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
# class ResNet_18_server_side(nn.Module):
#     def __init__(self, ResidualBlock, args, num_classes=100):
#         super(ResNet_18_server_side, self).__init__()
#         self.inchannel = 64
#         self.layers = nn.ModuleList()
#         if args.split_layer < 1:
#             self.layers.append(self.make_layer(ResidualBlock, 64, 2, stride=1))
#         if args.split_layer < 2:
#             self.layers.append(self.make_layer(ResidualBlock, 128, 2, stride=2))
#         if args.split_layer < 3:
#             self.layers.append(self.make_layer(ResidualBlock, 256, 2, stride=2))
#         if args.split_layer < 4:
#             self.layers.append(self.make_layer(ResidualBlock, 512, 2, stride=2))
#         self.fc = nn.Linear(512, num_classes)

#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         out = F.avg_pool2d(x, 4)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out
    

class ResNet_50_client_side(nn.Module):
    def __init__(self, args):
        super(ResNet_50_client_side, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.layers = nn.ModuleList()
        
        if args.split_layer >= 1:
            downsample_l1 = nn.Sequential(conv1x1(in_planes=64, out_planes=256, stride=1), norm_layer(256))
            self.layers.append(nn.Sequential(
                Bottleneck(inplanes=64, planes=64, stride=1, downsample=downsample_l1, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=256, planes=64, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
                Bottleneck(inplanes=256, planes=64, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                        
            ))
        
        if args.split_layer >= 2:
            downsample_l2 = nn.Sequential(conv1x1(in_planes=256, out_planes=512, stride=2), norm_layer(512))
            self.layers.append(nn.Sequential(
                Bottleneck(inplanes=256, planes=128, stride=2, downsample=downsample_l2, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
                Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),          
            ))
        
        if args.split_layer >= 3:
            downsample_l3 = nn.Sequential(conv1x1(in_planes=512, out_planes=1024, stride=2), norm_layer(1024))
            self.layers.append(nn.Sequential(
                Bottleneck(inplanes=512, planes=256, stride=2, downsample=downsample_l3, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),   
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                
            ))
        
        if args.split_layer >= 4:
            downsample_l4 = nn.Sequential(conv1x1(in_planes=1024, out_planes=2048, stride=2), norm_layer(2048))
            self.layers.append(nn.Sequential(
                Bottleneck(inplanes=1024, planes=512, stride=2, downsample=downsample_l4, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
                Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            ))
        
    def forward(self, x):
        out = self.initial_conv(x)
        
        for layer in self.layers:
            out = layer(out)
        
        return out

class ResNet_50_server_side(nn.Module):
    def __init__(self, args, num_classes=2):
        super(ResNet_50_server_side, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        
        self.layers = nn.ModuleList()
        
        if args.split_layer < 4:
            downsample_l4 = nn.Sequential(conv1x1(in_planes=1024, out_planes=2048, stride=2), norm_layer(2048))
            self.layers.append(nn.Sequential(
                Bottleneck(inplanes=1024, planes=512, stride=2, downsample=downsample_l4, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
                Bottleneck(inplanes=2048, planes=512, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
            ))
        
        if args.split_layer < 3:
            downsample_l3 = nn.Sequential(conv1x1(in_planes=512, out_planes=1024, stride=2), norm_layer(1024))
            self.layers.append(nn.Sequential(
                Bottleneck(inplanes=512, planes=256, stride=2, downsample=downsample_l3, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),   
                Bottleneck(inplanes=1024, planes=256, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                
            ))
        
        if args.split_layer < 2:
            downsample_l2 = nn.Sequential(conv1x1(in_planes=256, out_planes=512, stride=2), norm_layer(512))
            self.layers.append(nn.Sequential(
                Bottleneck(inplanes=256, planes=128, stride=2, downsample=downsample_l2, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
                Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=512, planes=128, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),          
            ))
        
        if args.split_layer < 1:
            downsample_l1 = nn.Sequential(conv1x1(in_planes=64, out_planes=256, stride=1), norm_layer(256))
            self.layers.append(nn.Sequential(
                Bottleneck(inplanes=64, planes=64, stride=1, downsample=downsample_l1, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),
                Bottleneck(inplanes=256, planes=64, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),            
                Bottleneck(inplanes=256, planes=64, stride=1, downsample=None, 
                           groups=1, base_width=64, dilation=1, norm_layer=norm_layer),                        
            ))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        ### Projection head
        num_ftrs = args.out_dim
        self.fc1 = nn.Linear(2048, num_ftrs)
        self.fc1_bn = nn.BatchNorm1d(num_ftrs)
        self.fc2 = nn.Linear(num_ftrs, num_ftrs)
        self.fc2_bn = nn.BatchNorm1d(num_ftrs)
        self.fc3 = nn.Linear(num_ftrs, num_classes)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        x = self.relu(self.fc1_bn(self.fc1(x)))
        x = self.relu(self.fc2_bn(self.fc2(x)))
        preds = self.fc3(x)

        return preds


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s, should_pool=True):
    layers = [conv_layer(in_list[0], out_list[0], k_list[0], p_list[0])]
    layers += [conv_layer(out_list[i-1], out_list[i], k_list[i], p_list[i]) for i in range(1, len(out_list))]
    if should_pool:
        layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16_client_side(nn.Module):
    def __init__(self, args):
        super(VGG16_client_side, self).__init__()
        self.layers = nn.ModuleList()
        
        if args.split_layer >= 1:
            self.layers.append(vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2))
        
        if args.split_layer >= 2:
            self.layers.append(vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2))
        
        if args.split_layer >= 3:
            self.layers.append(vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2))
        
        if args.split_layer >= 4:
            self.layers.append(vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2))
        
        if args.split_layer >= 5:
            self.layers.append(vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2, should_pool=False))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VGG16_server_side(nn.Module):
    def __init__(self, args, num_classes=1000):
        super(VGG16_server_side, self).__init__()
        self.layers = nn.ModuleList()
        
        if args.split_layer < 1:
            self.layers.append(vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2))

        if args.split_layer < 2:
            self.layers.append(vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2))
        
        if args.split_layer < 3:
            self.layers.append(vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2))

        if args.split_layer < 4:
            self.layers.append(vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2))
        
        if args.split_layer < 5:
            self.layers.append(vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2, should_pool=False))
        
        self.layer6 = vgg_fc_layer(512*2*2, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)
        self.layer8 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        vgg16_features = x
        out = x.view(x.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        
        return out