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
        
        assert 1 <= args.split_layer <= 4

        self.layers = nn.ModuleList()

        self.layers.append(self.make_layer(ResidualBlock, 64, 2, stride=1))

        if args.split_layer >= 2:
            self.layers.append(self.make_layer(ResidualBlock, 128, 2, stride=2))

        if args.split_layer >= 3:
            self.layers.append(self.make_layer(ResidualBlock, 256, 2, stride=2))

        if args.split_layer >= 4:
            self.layers.append(self.make_layer(ResidualBlock, 512, 2, stride=2))

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        for layer in self.layers:
            out = layer(out)
        return out


class ResNet_18_server_side(nn.Module):
    def __init__(self, ResidualBlock, args, num_classes=100):
        super(ResNet_18_server_side, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.inchannel = 64 * (2 ** (args.split_layer - 1))
            
        if args.split_layer < 2:
            self.layers.append(self.make_layer(ResidualBlock, 128, 2, stride=2))
        
        if args.split_layer < 3:
            self.layers.append(self.make_layer(ResidualBlock, 256, 2, stride=2))
        
        if args.split_layer < 4:
            self.layers.append(self.make_layer(ResidualBlock, 512, 2, stride=2))
        
        self.fc = nn.Linear(512, num_classes)
    
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = nn.functional.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    

class ResNet_50_client_side(nn.Module):
    def __init__(self, args):
        super(ResNet_50_client_side, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        assert 1 <= args.split_layer <= 4
        
        self.layers = nn.ModuleList()
        
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
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        ### Projection head
        if args.split_layer == 4:
            num_ftrs = 2048
        else:
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


from typing import Any, Callable, List, Optional, Union

import torch
from torch import nn, Tensor

__all__ = [
    "MobileNetV3",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
]


class SqueezeExcitation(torch.nn.Module):
    """This Squeeze-and-Excitation block
    Args:
        in_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
    """

    def __init__(
            self,
            in_channels: int,
            squeeze_channels: int,
    ) -> None:
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, in_channels, 1)
        self.relu = nn.ReLU()  # `delta` activation
        self.hard = nn.Hardsigmoid()  # `sigma` (aka scale) activation

    def forward(self, x: Tensor) -> Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hard(scale)
        return scale * x


def _make_divisible(v: float, divisor: int = 8) -> int:
    """This function ensures that all layers have a channel number divisible by 8"""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(torch.nn.Sequential):
    """Convolutional block, consists of nn.Conv2d, nn.BatchNorm2d, nn.ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional = None,
            groups: int = 1,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: bool = False,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.01)
        ]

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """Inverted Residual block"""

    def __init__(
            self,
            in_channels: int,
            kernel: int,
            exp_channels: int,
            out_channels: int,
            use_se: bool,
            activation: str,
            stride: int,
            dilation: int,
    ) -> None:
        super().__init__()
        self._shortcut = stride == 1 and in_channels == out_channels

        in_channels = _make_divisible(in_channels)
        exp_channels = _make_divisible(exp_channels)
        out_channels = _make_divisible(out_channels)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if activation == "HS" else nn.ReLU

        # expand
        if exp_channels != in_channels:
            layers.append(
                Conv2dNormActivation(
                    in_channels=in_channels,
                    out_channels=exp_channels,
                    kernel_size=1,
                    activation_layer=activation_layer,
                )
            )

        # depth-wise convolution
        layers.append(
            Conv2dNormActivation(
                in_channels=exp_channels,
                out_channels=exp_channels,
                kernel_size=kernel,
                stride=1 if dilation > 1 else stride,
                dilation=dilation,
                groups=exp_channels,
                activation_layer=activation_layer,
            )
        )
        if use_se:
            squeeze_channels = _make_divisible(exp_channels // 4, 8)
            layers.append(
                SqueezeExcitation(
                    in_channels=exp_channels,
                    squeeze_channels=squeeze_channels
                )
            )

        # project layer
        layers.append(
            Conv2dNormActivation(
                in_channels=exp_channels,
                out_channels=out_channels,
                kernel_size=1,
                activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self._shortcut:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[List[Union[int, str, bool]]],
                last_channel: int,
            num_classes: int = 1000,
            dropout: float = 0.2,
    ) -> None:
        """MobileNet V3 main class
        Args:
            inverted_residual_setting: network structure
            last_channel: number of channels on the penultimate layer
            num_classes: number of classes
            dropout: dropout probability
        """
        super().__init__()

        # building first layer
        first_conv_out_channels = inverted_residual_setting[0][0]
        layers: List[nn.Module] = [
            Conv2dNormActivation(
                in_channels=3,
                out_channels=first_conv_out_channels,
                kernel_size=3,
                stride=2,
                activation_layer=nn.Hardswish,
            )
        ]

        # building inverted residual blocks
        for params in inverted_residual_setting:
            layers.append(InvertedResidual(*params))

        # building last several layers
        last_conv_in_channels = inverted_residual_setting[-1][3]
        last_conv_out_channels = 6 * last_conv_in_channels
        layers.append(
            Conv2dNormActivation(
                in_channels=last_conv_in_channels,
                out_channels=last_conv_out_channels,
                kernel_size=1,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_out_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return 0, 0, x, 0, 0

def Get_Arch(arch):
    if arch == "mobilenet_v3_large":
        return [
            [16, 3, 16, 16, False, "RE", 1, 1],
            [16, 3, 64, 24, False, "RE", 2, 1],  # C1
            [24, 3, 72, 24, False, "RE", 1, 1],
            [24, 5, 72, 40, True, "RE", 2, 1],  # C2
            [40, 5, 120, 40, True, "RE", 1, 1],
            [40, 5, 120, 40, True, "RE", 1, 1],
            [40, 3, 240, 80, False, "HS", 2, 1],  # C3
            [80, 3, 200, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 480, 112, True, "HS", 1, 1],
            [112, 3, 672, 112, True, "HS", 1, 1],
            [112, 5, 672, 160, True, "HS", 2, 1],  # C4
            [160, 5, 960, 160, True, "HS", 1, 1],
            [160, 5, 960, 160, True, "HS", 1, 1],
        ]
    elif arch == "mobilenet_v3_small":
        return [
            [16, 3, 16, 16, True, "RE", 2, 1],  # C1
            [16, 3, 72, 24, False, "RE", 2, 1],  # C2
            [24, 3, 88, 24, False, "RE", 1, 1],
            [24, 5, 96, 40, True, "HS", 2, 1],  # C3
            [40, 5, 240, 40, True, "HS", 1, 1],
            [40, 5, 240, 40, True, "HS", 1, 1],
            [40, 5, 120, 48, True, "HS", 1, 1],
            [48, 5, 144, 48, True, "HS", 1, 1],
            [48, 5, 288, 96, True, "HS", 2, 1],  # C4
            [96, 5, 576, 96, True, "HS", 1, 1],
            [96, 5, 576, 96, True, "HS", 1, 1],
        ]
    
    
def _mobilenet_v3(arch: str, **kwargs: Any, ) -> MobileNetV3:
    inverted_residual_setting = Get_Arch(arch)
    if arch == "mobilenet_v3_large":
        last_channel = 1280  # C5
    elif arch == "mobilenet_v3_small":
        last_channel = 1024  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    return model


def MobileNetV3_(**kwargs: Any) -> MobileNetV3:
    return _mobilenet_v3(arch="mobilenet_v3_large", **kwargs)


def MobileNetV3_small(**kwargs: Any) -> MobileNetV3:
    return _mobilenet_v3(arch="mobilenet_v3_small", **kwargs)


class MobileNetV3_client_side(nn.Module):
    def __init__(self, args):
        super(MobileNetV3_client_side, self).__init__()
        # Always default to large
        inverted_residual_setting = Get_Arch("mobilenet_v3_large")
        first_conv_out_channels = inverted_residual_setting[0][0]
        self.features = nn.Sequential(
            Conv2dNormActivation(
                in_channels=3,
                out_channels=first_conv_out_channels,
                kernel_size=3,
                stride=2,
                activation_layer=nn.Hardswish,
            )
        )

        assert 1 <= args.split_layer <= 4

        self.layers = nn.ModuleList()
        start_idx = 0
        if args.split_layer >= 1:
            end_idx = len(inverted_residual_setting) // 4
            self.layers.append(self.make_layer(inverted_residual_setting, start_idx, end_idx))
            start_idx = end_idx
        if args.split_layer >= 2:
            end_idx = len(inverted_residual_setting) // 2
            self.layers.append(self.make_layer(inverted_residual_setting, start_idx, end_idx))
            start_idx = end_idx
        if args.split_layer >= 3:
            end_idx = len(inverted_residual_setting) * 3 // 4
            self.layers.append(self.make_layer(inverted_residual_setting, start_idx, end_idx))
            start_idx = end_idx
        if args.split_layer >= 4:
            end_idx = len(inverted_residual_setting)
            self.layers.append(self.make_layer(inverted_residual_setting, start_idx, end_idx))

    def make_layer(self, inverted_residual_setting, start_idx, end_idx):
        layers = []
        for params in inverted_residual_setting[start_idx:end_idx]:
            layers.append(InvertedResidual(*params))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MobileNetV3_server_side(nn.Module):
    def __init__(self, args, num_classes=1000):
        super(MobileNetV3_server_side, self).__init__()
        inverted_residual_setting = Get_Arch("mobilenet_v3_large")
        last_channel = 1280
        dropout = 0.2

        self.layers = nn.ModuleList()
        if args.split_layer < 4:
            start_idx = len(inverted_residual_setting) * 3 // 4
            end_idx = len(inverted_residual_setting)
            self.layers.append(self.make_layer(inverted_residual_setting, start_idx, end_idx))

        last_conv_in_channels = inverted_residual_setting[-1][3]
        last_conv_out_channels = 6 * last_conv_in_channels
        self.layers.append(
            Conv2dNormActivation(
                in_channels=last_conv_in_channels,
                out_channels=last_conv_out_channels,
                kernel_size=1,
                activation_layer=nn.Hardswish,
            )
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_out_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

    def make_layer(self, inverted_residual_setting, start_idx, end_idx):
        layers = []
        for params in inverted_residual_setting[start_idx:end_idx]:
            layers.append(InvertedResidual(*params))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x