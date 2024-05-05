import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AudioResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4, num_mels=128, use_etf=True):
        super(AudioResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.use_etf = use_etf
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize ETF classifier if use_etf is True
        if self.use_etf:
            self.init_etf(512 * block.expansion) 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def init_etf(self, out_dim):
        with torch.no_grad():
            # Create a base ETF matrix
            base_etf = torch.sqrt(torch.tensor(self.num_classes / (self.num_classes - 1))) * (
                torch.eye(self.num_classes) - (1 / self.num_classes) * torch.ones((self.num_classes, self.num_classes)))
            base_etf /= torch.sqrt((1 / self.num_classes * torch.norm(base_etf, 'fro') ** 2))
            
            # Random initialization for high-dimensional projection
            random_proj = torch.randn(out_dim, self.num_classes)

            # Apply the ETF transformation to the random projection matrix
            transformed_weights = torch.mm(random_proj, base_etf)

            # Set the weights of the linear layer
            self.linear.weight = nn.Parameter(transformed_weights.transpose(0, 1))
            self.linear.weight.requires_grad_(False)



    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
