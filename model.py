import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict
import pandas as pd
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def conv_block(in_channels, out_channels, pool = False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace = True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class AudioResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(AudioResNet, self).__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(OrderedDict([("conv1res1", conv_block(128, 128)), ("conv2res1", conv_block(128, 128))]))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(OrderedDict([("conv3res2", conv_block(512, 512)), ("conv4res2", conv_block(512, 512))]))
        
        self.conv5 = conv_block(512, 1024, pool=True)
        self.conv6 = conv_block(1024, 2048, pool=True)
        self.res3 = nn.Sequential(OrderedDict([("conv5res3", conv_block(2048, 2048)), ("conv6res3", conv_block(2048, 2048))]))
        
        self.conv7 = conv_block(2048, 4096, pool=True)
        self.conv8 = conv_block(4096, 8192, pool=True)
        self.res4 = nn.Sequential(OrderedDict([("conv5res3", conv_block(8192, 8192)), ("conv6res3", conv_block(8192, 8192))]))
        
        # self.conv9 = conv_block(8192, 16384, pool=True)
        # self.conv10 = conv_block(16384, 32768, pool=True)
        # self.res5 = nn.Sequential(OrderedDict([("conv5res3", conv_block(32768, 32768)), ("conv6res3", conv_block(32768, 32768))]))
        
#         self.conv11 = conv_block(32768, 65536)
#         self.conv12 = conv_block(65536, 131072, pool=True)
#         self.res6 = nn.Sequential(OrderedDict([("conv1res1", conv_block(131072, 131072)), ("conv2res1", conv_block(131072, 131072))]))
        
#         self.conv13 = conv_block(131072, 262144, pool=True)
#         self.conv14 = conv_block(262144, 262144, pool=True)
#         self.res7 = nn.Sequential(OrderedDict([("conv3res2", conv_block(262144, 262144)), ("conv4res2", conv_block(262144, 262144))]))
        
#         self.conv15 = conv_block(262144, 524288, pool=True)
#         self.conv16 = conv_block(524288, 1048576, pool=True)
#         self.res8 = nn.Sequential(conv_block(1048576, 1048576), conv_block(1048576, 1048576))
        
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.MaxPool2d((1, 1)),
                                       nn.Flatten(),
                                       nn.Linear(8192, num_classes))
                                       #nn.Linear(32768, num_classes))
#                                        nn.Linear(1048576, num_classes))

#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.linear = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.res4(out) + out
        # out = self.conv9(out)
        # out = self.conv10(out)
        # out = self.res5(out) + out
#         out = self.conv11(x)
#         out = self.conv12(out)
#         out = self.res6(out) + out
#         out = self.conv13(out)
#         out = self.conv14(out)
#         out = self.res7(out) + out
#         out = self.conv15(out)
#         out = self.conv16(out)
#         out = self.res8(out) + out
        
        return self.classifier(out)
        

#         out = self.adaptive_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out





# import torch
# import torch.nn as nn
# import torchvision.models as models

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes)
#             )
#         else:
#             self.shortcut = nn.Sequential()

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class AudioResNet(nn.Module):
#     def __init__(self, base_model='resnet50', num_classes=4, use_etf=False):
#         super(AudioResNet, self).__init__()
#         self.num_classes = num_classes
#         self.use_etf = use_etf
        
#         # Dynamically load the specified ResNet model
#         if base_model == 'resnet34':
#             self.base_model = models.resnet34(pretrained=False)
#         elif base_model == 'resnet50':
#             self.base_model = models.resnet50(pretrained=False)
#         elif base_model == 'resnet101':
#             self.base_model = models.resnet101(pretrained=False)
#         else:
#             raise ValueError("Unsupported base model. Choose from 'resnet34', 'resnet50', or 'resnet101'.")

#         # Adapt the first convolution layer for single-channel input
#         self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
#         # Modify the output layer to fit the number of classes
#         num_ftrs = self.base_model.fc.in_features
#         self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        
#         # Initialize ETF classifier if use_etf is True
#         if self.use_etf:
#             self.init_etf(num_ftrs)

#     def init_etf(self, out_dim):
#         with torch.no_grad():
#             base_etf = torch.sqrt(torch.tensor(self.num_classes / (self.num_classes - 1))) * (
#                 torch.eye(self.num_classes) - (1 / self.num_classes) * torch.ones((self.num_classes, self.num_classes)))
#             base_etf /= torch.sqrt((1 / self.num_classes * torch.norm(base_etf, 'fro') ** 2))
#             random_proj = torch.randn(out_dim, self.num_classes)
#             transformed_weights = torch.mm(random_proj, base_etf)
#             self.base_model.fc.weight = nn.Parameter(transformed_weights.transpose(0, 1))
#             self.base_model.fc.weight.requires_grad_(False)

#     def forward(self, x):
#         return self.base_model(x)

# # # Example of usage
# # model_resnet50 = AudioResNet(base_model='resnet50', num_classes=10, use_etf=True)
# # model_resnet34 = AudioResNet(base_model='resnet34', num_classes=10)
# # model_resnet101 = AudioResNet(base_model='resnet101', num_classes=10)
