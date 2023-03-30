import torch
import torch.nn as nn

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RepVGGBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels == out_channels and stride == 1:
            self.identity = nn.Identity()
        else:
            self.identity = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out + self.identity(x)
        return out

class Net(nn.Module):
    def __init__(self, num_classes=752, reid=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(RepVGGBlock(64, 64, 1), RepVGGBlock(64, 64, 1), RepVGGBlock(64, 128, 2),
                                    RepVGGBlock(128, 128, 1), RepVGGBlock(128, 128, 1), RepVGGBlock(128, 256, 2),
                                    RepVGGBlock(256, 256, 1), RepVGGBlock(256, 256, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.reid = reid
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.blocks(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if self.reid:
            out = out.div(out.norm(p=2, dim=1, keepdim=True))
            return out
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    net = Net()
    x = torch.randn(4, 3, 128, 64) # [batch size, channel, height, width]
    y = net.forward(x)
    print(y)