import pickle
import json
import os
import models
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from numpy.linalg import norm
from sklearn.model_selection import train_test_split


torch.backends.cudnn.benchmark=True
torch.manual_seed(42)
np.random.seed(42)


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
    dataset = get_parameter(param="dataset")
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
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
    return loss.item()


def weight_aggregate(local_model, client_models, client_gradients, client_idx):
    for i in client_idx:
        model = client_models[i]
        w = model.state_dict()['fc.weight'].cpu().numpy()
        global_w = local_model.state_dict()['fc.weight'].cpu().numpy()
        gradient = global_w - w
        client_gradients[i] =  np.add(client_gradients[i], gradient)
    return client_gradients


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
    threshold = get_parameter(param="threshold")
    delta = get_parameter(param="delta")
    delta = get_parameter(param="delta")

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


def aggregate(local_model, client_models, phi, client_idx):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = local_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[idx].state_dict()[k].float()* phi[idx] for idx in client_idx], 0).mean(0)
    local_model.load_state_dict(global_dict)

    return local_model


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
    dataset = get_parameter(param="dataset")
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
    dataset = get_parameter(param="dataset")
    batch_size = get_parameter(param="batch_size")

    if dataset == "MNIST":
        transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])
        
        testdata = datasets.MNIST('./data/test/', train=False, transform=transform, download=True)
        
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
                    './data',
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
    dataset = get_parameter(param="dataset")
    num_clients = get_parameter(param="num_clients")

    if dataset == "MNIST":
        lr = 0.01
        local_model =  models.SFMNet(784, 10)
        client_gradients = np.zeros([num_clients, 10, 784])
        client_models = [models.SFMNet(784, 10) for _ in range(num_clients)]
    elif dataset == "CIFAR":
        lr = 0.1
        local_model =  models.VGG('VGG19')
        client_gradients = np.zeros([num_clients, 10, 512])   # dimensions of the last layer
        client_models = [models.VGG('VGG19') for _ in range(num_clients)]
    elif dataset == "KDD":
        lr = 0.001
        local_model =  models.SFMNet(n_features= 41, n_classes= 23)
        client_gradients = np.zeros([num_clients, 23, 41])   # dimensions of the last layer
        client_models = [models.SFMNet(n_features= 41, n_classes= 23) for _ in range(num_clients)]


    # cosine-similarity matrix
    cs_mat = np.zeros((num_clients, num_clients), dtype=float) * 1e-6

    # trust scores
    r = [1 / num_clients for _ in range(num_clients)]  

    for _, model in enumerate(client_models):
        model.load_state_dict(local_model.state_dict())

    opt = [optim.SGD(model.parameters(), lr=lr) for model in client_models]

    return local_model, client_models, client_gradients, opt, cs_mat, r


def run_fl(local_model, client_models, opt, r, cs_mat, client_gradients, alpha="100", attack_type="lf"):
    if attack_type not in ["backdoor", "lf"]:
        print("[x] ERROR: attack type not recognized :)")
        return

    if alpha not in ["100","0.05","10"]:
        print("[x] ERROR: alpha not recognized :)")
        return

    dataset = get_parameter(param="dataset")
    num_rounds = get_parameter(param="num_rounds")
    num_clients = get_parameter(param="num_clients")
    num_selected = get_parameter(param="num_selected")
    epochs = get_parameter(param="epochs")
    batch_size = get_parameter(param="batch_size")

    clients = np.arange(num_clients)
    dishonest_client_idx = np.random.permutation(num_clients)[:int(0.9 * num_selected)]

    attack = 0
    for round in range(num_rounds):
        probs = [r[i] / sum(r) for i in range(num_clients)]
        client_idx = np.random.choice(clients, size=num_selected, replace=False, p=probs)
        
        # client update
        loss = 0
        print(round)
        for j, i in tqdm(enumerate(client_idx)):
            # get the global weights
            client_models[i].load_state_dict(local_model.state_dict())
            
            # read the local data
            if dataset == "MNIST" or dataset == "CIFAR":
                train_obj = pickle.load(open("./data/" + dataset + "/" + str(i) + "/train_" + alpha +"_.pickle", "rb"))
                x = torch.stack(train_obj.x)
                y = torch.tensor(train_obj.y)
                dat = TensorDataset(x, y)
                train_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)
                if j in dishonest_client_idx:
                    loss += client_update(client_model=client_models[i], optimizer=opt[i], train_loader=train_loader, epoch=epochs, attack_type=attack_type)
                else:
                    loss += client_update(client_model=client_models[i], optimizer=opt[i], train_loader=train_loader, epoch=epochs)
            elif dataset == "KDD":
                _, train_x, train_y = load_data(dataset="KDD")
                x = torch.tensor(train_x[int(i* len(train_x)/num_clients):int((i+1)*len(train_x)/num_clients)])
                y = torch.tensor(train_y[int(i* len(train_x)/num_clients):int((i+1)*len(train_x)/num_clients)])
                dat = TensorDataset(x, y)
                train_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)
                
                # honest or not honest update
                if j in dishonest_client_idx:
                    attack += 1
                    loss += client_update(client_models[i], opt[i], train_loader, epoch=epochs)
                else:
                    loss += client_update(client_models[i], opt[i], train_loader, epoch=epochs)

        local_model, r, cs_mat, client_gradients = compute_global_parameters(
            local_model=local_model,
            r=r,
            cs_mat=cs_mat,
            client_gradients=client_gradients,
            client_models=client_models,
            client_idx=client_idx
        )

    return loss, attack


def compute_global_parameters(local_model, client_models, client_idx, r, cs_mat, client_gradients):
    num_clients = get_parameter(param="num_clients")

    # historical gradient aggregate.
    client_gradients = weight_aggregate(local_model, client_models, client_gradients, client_idx)
    
    # compute similarity matrix and alignment scores scores.
    cs_mat = compute_cosine_sim(client_gradients, cs_mat)
    
    # pardoning the honest clients
    cs_mat = pardoning_fun(cs_mat)
    
    # alignment scores
    client_scores = np.mean(cs_mat, axis=1)

    # compute and normalize trust scores
    r = compute_trust_scores(client_scores, r)
    r = [r[i] / max(r) for i in range(num_clients)]

    # update the contribution rates
    phi = compute_contribitions(client_scores)

    # server aggregate
    #print(client_scores)
    local_model = aggregate(local_model, client_models, phi, client_idx)

    return local_model, r, cs_mat, client_gradients


def evaluate(local_model, loss, attack=0):
    losses_test = []
    acc_test = []

    num_selected = get_parameter(param="num_selected")
    batch_size = get_parameter(param="batch_size")

    test_loader, _, _ = load_data()
    test_loss, acc, attack = test(local_model=local_model, test_loader=test_loader, attack=attack)
    losses_test.append(test_loss)
    acc_test.append(acc)

    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))
    asr = attack/(len(test_loader)*batch_size)
    print('attack success rate %0.3g' %(asr))


def get_parameter(param: str):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config[param]
