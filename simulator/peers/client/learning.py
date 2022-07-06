import asyncio
import pickle
import os
import models
import utils

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from os.path import exists
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from models import VGG, SFMNet

torch.backends.cudnn.benchmark=True
torch.manual_seed(42)
np.random.seed(42)


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


def aggregate(peers_indices, peers_weights=[]):
    phi = utils.get_phi()
    global_dict = peers_weights[-1].state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([peers_weights[idx].state_dict()[k].float()* phi[peers_indices[idx]] for idx in range(0, len(peers_indices))], 0).mean(0)
    
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
        
        testdata = datasets.MNIST(os.getenv("DATA_FOLDER") + 'test/', train=False, transform=transform, download=True)
        
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
                    os.getenv("DATA_FOLDER"),
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


def initialize(modelID):
    dataset = utils.get_parameter(param="dataset")

    if dataset == "MNIST":
        local_model =  models.SFMNet(784, 10)
    elif dataset == "CIFAR":
        local_model =  models.VGG('VGG19')
    elif dataset == "KDD":
        local_model =  models.SFMNet(n_features= 41, n_classes= 23)

    if not exists(os.getenv("TMP_FOLDER") + modelID + ".pt"):
        torch.save(local_model.state_dict(), os.getenv("TMP_FOLDER") + modelID + ".pt")

    return local_model


def evaluate(local_model, loss, attack=0):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    losses_test = []
    acc_test = []

    num_selected = utils.get_parameter(param="num_selected")
    batch_size = utils.get_parameter(param="batch_size")

    test_loader, _, _ = load_data()
    test_loss, acc, attack = test(local_model=local_model, test_loader=test_loader, attack=attack)
    losses_test.append(test_loss)
    acc_test.append(acc)

    message = 'average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc)
    print(message)
    loop.run_until_complete(utils.send_log(message))
    asr = attack/(len(test_loader)*batch_size)
    message = 'attack success rate %0.3g' %(asr)
    print(message)
    loop.run_until_complete(utils.send_log(message))
    return acc, asr


def train(local_model, opt, peers_weights, peers_indices, dishonest_peers=[], alpha="100", attack_type="lf"):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if attack_type not in ["backdoor", "lf"]:
        message = "[x] ERROR: attack type not recognized :)"
        print(message)
        loop.run_until_complete(utils.send_log(message))
        return

    if alpha not in ["100","0.05","10"]:
        message = "[x] ERROR: alpha not recognized :)"
        print(message)
        loop.run_until_complete(utils.send_log(message))
        return

    dataset = utils.get_parameter(param="dataset")
    num_clients = utils.get_parameter(param="num_clients")
    epochs = utils.get_parameter(param="epochs")
    batch_size = utils.get_parameter(param="batch_size")
    
    weights = []

    for w in peers_weights:
        # w = utils.get_weights(path=path)
        m = load_weights_into_model(w)
        weights.append(m)
       
    weights.append(local_model)
    peers_indices.append(int(os.getenv("MY_ID")))
    local_model = aggregate(peers_weights=weights, peers_indices=peers_indices)

    attack = 0

    # client update
    loss = 0

    my_id = int(os.getenv("MY_ID"))

    # read the local data
    if dataset == "MNIST" or dataset == "CIFAR":
        train_obj = pickle.load(open(os.getenv("DATA_FOLDER") + dataset + "/" + str(my_id) + "/train_" + alpha +"_.pickle", "rb"))
        x = torch.stack(train_obj.x)
        y = torch.tensor(train_obj.y)
        dat = TensorDataset(x, y)
        train_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)
    elif dataset == "KDD":
        _, train_x, train_y = load_data()
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


def load_weights_into_model(weights):
    dataset = utils.get_parameter(param="dataset")
    if dataset == "MNIST":
        local_model =  models.SFMNet(784, 10)
        state_dict = local_model.state_dict()
        state_dict['fc.weight'] = weights['fc.weight']
        state_dict['fc.bias'] = weights['fc.bias']
        local_model.load_state_dict(state_dict)
    elif dataset == "CIFAR":
        local_model =  models.VGG('VGG19')
        state_dict = local_model.state_dict()
        state_dict['fc.weight'] = weights['fc.weight']
        state_dict['fc.bias'] = weights['fc.bias']
        local_model.load_state_dict(state_dict)
    elif dataset == "KDD":
        local_model =  models.SFMNet(n_features= 41, n_classes= 23)
        state_dict = local_model.state_dict()
        state_dict['fc.weight'] = weights['fc.weight']
        state_dict['fc.bias'] = weights['fc.bias']
        local_model.load_state_dict(state_dict)

    return local_model


def get_optimizer():
    dataset = utils.get_parameter(param="dataset")
    if dataset == "MNIST":
        lr = 0.01
        model = SFMNet(784, 10)
        opt = optim.SGD(model.parameters(), lr=lr)
    elif dataset == "CIFAR":
        lr = 0.1
        model = VGG('VGG19')
        opt = optim.SGD(model.parameters(), lr=lr)
    elif dataset == "KDD":
        lr = 0.001
        model = SFMNet(n_features= 41, n_classes= 23)
        opt = optim.SGD(model.parameters(), lr=lr)
    return opt


def load_local_model():
    dataset = utils.get_parameter(param="dataset")
    model = None
    if dataset == "MNIST":
        model = SFMNet(784, 10)
    elif dataset == "CIFAR":
        model = VGG('VGG19')
    elif dataset == "KDD":
        model = SFMNet(n_features= 41, n_classes= 23)
    return model


def learn(modelID):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    print(os.getenv("MY_NAME"), "Learning")
    loop.run_until_complete(utils.send_log("Learning"))
    if not exists(os.getenv("TMP_FOLDER") + modelID + ".pt"):
        local_model = initialize(modelID)
    else:
        local_model = load_local_model()
        local_model.load_state_dict(torch.load(os.getenv("TMP_FOLDER") + modelID + ".pt"))
    
    weights, indices, parents = utils.get_weights_to_train(modelID=modelID)

    message = "Weights to Train From = " + str(len(weights))
    print(message)
    loop.run_until_complete(utils.send_log(message))
    if len(weights) > 0:
        print(os.getenv("MY_NAME"), "Training")
        loss, attack, local_model = train(
            local_model=local_model,
            opt=get_optimizer(),
            alpha=utils.get_parameter(param="alpha"),
            attack_type=utils.get_parameter(param="attack_type"),
            peers_indices=indices,
            peers_weights=weights,
        )
        utils.clear_weights_ids(modelID)

        print(os.getenv("MY_NAME"), "Evaluating")
        loop.run_until_complete(utils.send_log("Evaluating"))
        acc, asr = evaluate(local_model=local_model, loss=loss, attack=attack)    
        
        if acc >= utils.get_my_latest_accuracy():
            print(os.getenv("MY_NAME"), "Publishing")
            loop.run_until_complete(utils.send_log("Publishing"))
            torch.save(local_model.state_dict(), os.getenv("TMP_FOLDER") + modelID + ".pt")
            messageID = utils.publish_model_update(
                modelID=modelID,
                weights=local_model.state_dict()['fc.weight'].cpu().numpy(),
                accuracy=acc,
                parents=parents,
                model=local_model.state_dict()
            )
            print(os.getenv("MY_NAME"), acc, messageID)
            utils.store_my_latest_accuracy(accuracy=acc)
            loop.run_until_complete(utils.send_log(str(acc) + " " + str(messageID)))
    else:
        print(os.getenv("MY_NAME"), "No weights to train from!")
        loop.run_until_complete(utils.send_log("No weights to train from!"))
