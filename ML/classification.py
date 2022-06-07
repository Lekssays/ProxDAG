import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD
from utils.model.NN import SFMNet
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18 as resent
from utils.data.load_dataset import load_torch_dataset, load_dataset

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        features = torch.tensor(self.x[idx], dtype=torch.float)
        targets = torch.tensor((self.y[idx]))
        return features,targets


class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.training_data = None
        self.training_target = None
        self.testing_data = None
        self.testing_target = None

        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.device = torch.device("cpu")

        self.initialize()


    def initialize(self):
        train_kwargs = {'batch_size': self.args.batch_size}
        test_kwargs = {'batch_size': self.args.test_batch_size}

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            cuda_kwargs = {'num_workers': self.args.workers,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        self.model = SFMNet(n_features=41, n_classes=23)
        self.optimizer = SGD(self.model.parameters(), lr=self.args.learning_rate)

        if self.args.dataset == 'MNIST':
            train_dataset, test_dataset = load_torch_dataset(self.args.dataset)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        elif self.args.dataset == 'kddcup':
            self.training_data, self.testing_data, self.training_target, self.testing_target = load_dataset('kddcup')
            self.train_loader = DataLoader(Dataset(self.testing_data, self.testing_target), **train_kwargs)
            self.test_loader =  DataLoader(Dataset(self.testing_data, self.testing_target), **test_kwargs)

        elif self.args.dataset == 'cifar':
            self.model = resent(num_classes=10)
            self.optimizer = SGD(self.model.parameters(), lr=self.args.learning_rate,
                                 momentum=0.9, weight_decay=1e-4)
            train_dataset, test_dataset = load_torch_dataset(self.args.dataset)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        self.model.to(self.device)


    def train(self):
        print("\nTraining...")
        for epoch in (range(self.args.iterations+1)):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                if self.args.dataset in ['MNIST', 'kddcup']:
                    loss = F.nll_loss(output, target)
                else:
                    loss = nn.CrossEntropyLoss(output, target)

                loss.backward()
                self.optimizer.step()

                if batch_idx % self.args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.item()))

        print('\nFinished training the classifier ')



    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if self.args.dataset in ['MNIST', 'kddcup']:
                    test_loss += F.nll_loss(output, target).sum(-1).item()
                else:
                    test_loss += nn.CrossEntropyLoss(output, target).sum(-1).item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))




# import cv2 as cv
# imagefile = self.dataDir+'/train-images-idx3-ubyte'
# cv.imshow("Image", cv.resize(imagearray[4], (760, 540)))
# cv.waitKey(3000);

