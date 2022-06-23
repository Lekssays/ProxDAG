import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as resent

#################################
##### Neural Network models #####
#################################

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

torch.manual_seed(42)
np.random.seed(42)


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = x.float()
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

torch.manual_seed(42)
np.random.seed(42)

class SFMNet(nn.Module):
    def __init__(self, n_features = 784, n_classes=10):
        super(SFMNet, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = x.float()
        x = x.view(-1, self.n_features)  # flatten
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class loan_model(nn.Module):
    def __init__(self, n_features = 127, n_classes=3):
        super(loan_model, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(True),
            nn.Linear(512, 41),
            nn.ReLU(True),
            nn.Linear(41, n_classes)
        )

    def forward(self, x):
        x = x.float()
        x = x.view(-1, self.n_features)  # flatten
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
