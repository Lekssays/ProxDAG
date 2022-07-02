import pickle
import os
import models
import utils

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from numpy.linalg import norm
from sklearn.model_selection import train_test_split


torch.backends.cudnn.benchmark=True
torch.manual_seed(42)
np.random.seed(42)

initialized = False

class client:
    def __init__(self, client_id, x, y):
        self.client_id = client_id
        self.x= x
        self.y=y

    def write_out(self, dataset, alpha, mode='train'):
        if not os.path.isdir(dataset):
            os.mkdir(dataset)
        if not os.path.isdir(dataset+'/'+str(self.client_id)):
            os.mkdir(dataset+'/'+str(self.client_id))

        with open(dataset+'/'+str(self.client_id)+'/'+mode+'_'+str(alpha)+'_'+'.pickle', 'wb') as f:
            pickle.dump(self, f)


def client_update(client_model, optimizer, train_loader, epoch=5, attack_type=None):
    """
    This function updates/trains client model on client data
    """
    dataset = utils.get_parameter(param="dataset")
    client_model.train()
    for _ in range(epoch):
        for _, (data, target) in enumerate(train_loader):
            data, target = data, target
            optimizer.zero_grad()
            output = client_model(data)
            if attack_type is not None:
                if dataset == "MNIST" or dataset == "CIFAR":
                    for i, t in enumerate(target):
                        if attack_type == 'lf':  # label flipping
                            if t == 1:
                                target[i] = torch.tensor(7)
                        elif attack_type == 'backdoor':
                            target[i] = 1  # set the label
                            data[:, :, 27, 27] = torch.max(data)  # set the bottom right pixel to white.
                        elif attack_type == "untargeted":
                            target[i] = 0
                elif dataset == "KDD":
                    for i, t in enumerate(target):
                        if attack_type == 'lf':  # label flipping
                            if t == 5:
                                target[i] = torch.tensor(7)
                        elif attack_type == 'backdoor':
                            pass
                        elif attack_type == "untargeted":
                            target[i] = 0
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item(), client_model


def compute_cosine_sim(client_gradients, cs_mat):
    # compute cosine similarities matrix
    for j, w1 in enumerate(client_gradients):
        for k, w2 in enumerate(client_gradients):
            if j == k:
                continue
            if np.all(w1==0) or np.all(w2==0):
                cs_mat[j][k] = 0
            else:
                cs_mat[j][k] = (w1 * w2).sum()/(norm(w1)*norm(w2))
    return cs_mat


def pardoning_fun(cs_mat):
    # pardoning function
    client_scores = np.mean(cs_mat, axis=1)
    for j, score1 in enumerate(client_scores):
        for k, score2 in enumerate(client_scores):
            if score2 == 0 and score1 > 0:
                scale = 1
            elif score2 == 0 and score1<=0:
                scale = 0
            else:
                scale = min(1, abs(score1 / score2))
            cs_mat[j][k] *= scale

    return cs_mat


def compute_trust_scores(client_scores, r):
    threshold = utils.get_parameter(param="threshold")
    delta = utils.get_parameter(param="delta")
    delta = utils.get_parameter(param="delta")

    # compute/update trust scores
    for i in range(len(client_scores)):
        #r[i] -= delta
        if client_scores[i] > threshold:
            r[i] -= delta
            r[i] = max(1e-6, r[i])
            continue
        r[i] += delta
    return r


def compute_contribitions(client_scores):
    phi = np.array([1 - client_scores[i] for i in range(len(client_scores))])
    phi[phi > 1] = 1;
    phi[phi < 0] = 0
    # Rescale so that max value is wv
    phi = phi / np.max(phi)
    phi[(phi == 1)] = .99
    # Logit function
    phi = (np.log((phi / (1 - phi)) + 0.000001) + 0.5)
    # phi[(np.isinf(phi))] = 1
    #return [phi[i] / max(phi) for i in range(len(client_scores))]
    phi[(np.isinf(phi) + phi > 1)] = 1
    phi[(phi < 0)] = 0
    return phi


def aggregate(peers_indices, peers_weights=[]):
    phi = utils.get_phi()
    global_dict = peers_weights[-1].state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([peers_weights[idx].state_dict()[k].float()* phi[idx] for idx in peers_indices], 0).mean(0)
    
    peers_weights[-1].load_state_dict(global_dict)
    return peers_weights[-1]


def load_kdd_dataset():
    kdd_df = pd.read_csv('kddcup.csv', delimiter=',', header=None)
    col_names = [str(i) for i in range(42)]
    kdd_df.columns=  col_names
    values_1 = kdd_df['1'].unique()
    values_2 = kdd_df['2'].unique()
    values_3 = kdd_df['3'].unique()
    targets = kdd_df['41'].unique()

    for i, v in enumerate(values_1):
        kdd_df.loc[kdd_df['1'] == v, '1'] = i
    for i, v in enumerate(values_2):
        kdd_df.loc[kdd_df['2'] == v, '2'] = i
    for i, v in enumerate(values_3):
        kdd_df.loc[kdd_df['3'] == v, '3'] = i
    for i, v in enumerate(targets):
        b = np.zeros(len(targets))
        b[i] = 1
        kdd_df.loc[kdd_df['41'] == v, '41'] = i

    for column in kdd_df.columns:
        kdd_df[column] = pd.to_numeric(kdd_df[column])
    y = np.array(kdd_df.iloc[:,-1].values)
    x = np.array(kdd_df.iloc[:,0:-1].values)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def test(local_model, test_loader, attack):
    """This function test the global model on test data and returns test loss and test accuracy """
    local_model.eval()
    test_loss = 0
    correct = 0
    dataset = utils.get_parameter(param="dataset")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = local_model(data)
            
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

            if dataset == "MNIST" or dataset == "CIFAR":
                for i, t in enumerate(target):
                    if t == 1 and pred[i] == 7:
                        attack += 1
            elif dataset == "KDD":
                for i, t in enumerate(target):
                    if t == 5 and pred[i] == 7:
                        attack += 1

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc, attack


def load_data():
    train_x, train_y, test_loader = None, None, None
    dataset = utils.get_parameter(param="dataset")
    batch_size = utils.get_parameter(param="batch_size")

    if dataset == "MNIST":
        transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
        
        testdata = datasets.MNIST('./../data/test/', train=False, transform=transform, download=True)
        
        # Loading the test data and thus converting them into a test_loader
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True)
    elif dataset == "CIFAR":
        # Image augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Loading the test iamges and thus converting them into a test_loader
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
                    './../data',
                    train=False,
                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                    batch_size=batch_size,
                    shuffle=True
                )
    elif dataset == "KDD":
        train_x, test_x, train_y, test_y = load_kdd_dataset()

        # Loading the test data and thus converting them into a test_loader
        x = torch.tensor(test_x)
        y = torch.tensor(test_y)
        np.random.seed(42)
        p = np.random.permutation(len(x))
        x = x[p]
        y= y[p]
        dat = TensorDataset(x, y)
        test_loader = torch.utils.data.DataLoader(dat,  batch_size=batch_size, shuffle=True)

    return test_loader, train_x, train_y


def initialize():
    dataset = utils.get_parameter(param="dataset")
    num_clients = utils.get_parameter(param="num_clients")

    if dataset == "MNIST":
        lr = 0.01
        local_model =  models.SFMNet(784, 10)
        peers_weights = [models.SFMNet(784, 10) for _ in range(num_clients)]
    elif dataset == "CIFAR":
        lr = 0.1
        local_model =  models.VGG('VGG19')
        peers_weights = [models.VGG('VGG19') for _ in range(num_clients)]
    elif dataset == "KDD":
        lr = 0.001
        local_model =  models.SFMNet(n_features= 41, n_classes= 23)
        peers_weights = [models.SFMNet(n_features= 41, n_classes= 23) for _ in range(num_clients)]

    for _, model in enumerate(peers_weights):
        model.load_state_dict(local_model.state_dict())

    opt = [optim.SGD(model.parameters(), lr=lr) for model in peers_weights]

    return local_model, peers_weights, opt


def evaluate(local_model, loss, attack=0):
    losses_test = []
    acc_test = []

    num_selected = utils.get_parameter(param="num_selected")
    batch_size = utils.get_parameter(param="batch_size")

    test_loader, _, _ = load_data()
    test_loss, acc, attack = test(local_model=local_model, test_loader=test_loader, attack=attack)
    losses_test.append(test_loss)
    acc_test.append(acc)

    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))
    asr = attack/(len(test_loader)*batch_size)
    print('attack success rate %0.3g' %(asr))

    return acc, asr


def train(local_model, opt, peers_weights, peers_indices, dishonest_peers=[], alpha="100", attack_type="lf"):
    if attack_type not in ["backdoor", "lf"]:
        print("[x] ERROR: attack type not recognized :)")
        return

    if alpha not in ["100","0.05","10"]:
        print("[x] ERROR: alpha not recognized :)")
        return

    dataset = utils.get_parameter(param="dataset")
    num_clients = utils.get_parameter(param="num_clients")
    epochs = utils.get_parameter(param="epochs")
    batch_size = utils.get_parameter(param="batch_size")

    weights = peers_weights.append(local_model)
    indices = peers_indices.append(int(os.getenv("MY_ID")))
    local_model = aggregate(peers_weights=weights, peers_indices=indices)

    attack = 0

    # client update
    loss = 0

    my_id = int(os.getenv("MY_ID"))

    # read the local data
    if dataset == "MNIST" or dataset == "CIFAR":
        train_obj = pickle.load(open("./../data/" + dataset + "/" + str(my_id) + "/train_" + alpha +"_.pickle", "rb"))
        x = torch.stack(train_obj.x)
        y = torch.tensor(train_obj.y)
        dat = TensorDataset(x, y)
        train_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)

    elif dataset == "KDD":
        _, train_x, train_y = load_data(dataset="KDD")
        x = torch.tensor(train_x[int(my_id* len(train_x)/num_clients):int((my_id+1)*len(train_x)/num_clients)])
        y = torch.tensor(train_y[int(my_id* len(train_x)/num_clients):int((my_id+1)*len(train_x)/num_clients)])
        dat = TensorDataset(x, y)
        train_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)

    if int(os.getenv("MY_ID")) in dishonest_peers:
        attack += 1
        loss, local_model = client_update(client_model=local_model, optimizer=opt, train_loader=train_loader, epoch=epochs, attack_type=attack_type)
    else:
        loss, local_model = client_update(client_model=local_model, optimizer=opt, train_loader=train_loader, epoch=epochs)

    return loss, attack, local_model


def learn(modelID: str):
    global initialized
    if not initialized:
        local_model, _, opt = initialize()
        initialized = True

    weights, indices, parents = utils.get_weights_to_train(modelID=modelID)

    loss, attack, local_model = train(
        local_model=local_model,
        opt=opt,
        alpha=utils.get_parameter(param="alpha"),
        attack_type=utils.get_parameter(param="attack_type"),
        peers_indices=indices,
        peers_weights=weights,
    )

    acc, asr = evaluate(local_model=local_model, loss=loss, attack=attack)

    if acc >= utils.get_my_latest_accuracy():
        utils.publish_model_update(
            modelID=modelID,
            weights=local_model.state_dict()['fc.weight'].cpu().numpy(),
            accuracy=acc,
            parents=parents,
        )
