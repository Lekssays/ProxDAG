import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # flatten
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
