import torch.nn as nn
import torch.nn.functional as F


class SFMNet(nn.Module):
    def __init__(self, n_features = 784, n_classes=10):
        super(SFMNet, self).__init__()
        self.n_features = n_features
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = x.view(-1, self.n_features)  # flatten
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


