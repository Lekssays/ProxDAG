import torch
import torch.nn.functional as F
import syft as sy

from torch.optim import SGD
from utils.model.NN import MnistNet
from utils.data.load_dataset import load_dataset
from utils.params.param_parser import parameter_parser


args = parameter_parser()


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.clients = ()  # tuple of clients/workers
        self.device = torch.device("cpu")

        self.initialize()


    def initialize(self):
        self.model = MnistNet()
        self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate)

        hook = sy.TorchHook(torch)
        train_kwargs = {'batch_size': args.batch_size, 'shuffle':True}
        test_kwargs = {'batch_size': args.test_batch_size, 'shuffle':True}

        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            cuda_kwargs = {'num_workers': args.workers,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)


        train_dataset, test_dataset = load_dataset()

        for i in range(1, args.clients_n+1):
            self.clients += (sy.VirtualWorker(hook, id="cs"+str(i)),)

        self.train_loader = sy.FederatedDataLoader(train_dataset.federate(self.clients), **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)


    def train(self):
        print("\nTraining...")
        for epoch in (range(args.iterations)):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # send the model to the remote location
                self.model = self.model.send(data.location)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                self.model.get() # get back the updated model.

                if batch_idx % args.log_interval == 0:
                    loss = loss.get()
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), self.train_loader.__len__(),
                               100. * batch_idx / len(self.train_loader), loss.item()))


        print('\nFinished Training the classifier ')



    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
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

