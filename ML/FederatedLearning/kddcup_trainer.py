###############################
##### importing libraries #####
###############################
import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models import SFMNet
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from numpy.linalg import norm
torch.backends.cudnn.benchmark=True

torch.manual_seed(42)
np.random.seed(42)

delta = 0.1
threshold = 0.1

attack = 0
asr = []


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


def load_dataset():

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


def client_update(client_model, optimizer, train_loader, epoch=5, honest= True, attack_type = 'lf'):
    """
    This function updates/trains client model on client data
    """
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            if not honest:
                for i, t in enumerate(target):
                    if attack_type == 'lf':  # label flipping
                        if t == 5:
                            target[i] = torch.tensor(7)
                    elif attack_type == 'backdoor':
                        pass
                    else:
                        target[i] = 0
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def weight_aggregate(global_model, client_models, client_gradients, client_idx):
    for i in client_idx:
        model = client_models[i]
        w = model.state_dict()['fc.weight'].cpu().numpy()
        global_w = global_model.state_dict()['fc.weight'].cpu().numpy()
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
    #print(client_scores)
    # compute/update trust scores
    for i in range(len(client_scores)):
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


def server_aggregate(global_model, client_models, phi, client_idx):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[idx].state_dict()[k].float()* phi[i] for idx in client_idx], 0).mean(0)
    global_model.load_state_dict(global_dict)


def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    global attack
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            for i, t in enumerate(target):
                if t == 5 and pred[i] == 7:
                    attack += 1

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

##### Hyperparameters for federated learning #########
num_clients = 300
num_selected = 30
num_rounds = 79
epochs = 3
batch_size = 50

#############################################################
##### Creating desired data distribution among clients  #####
#############################################################

train_x, test_x, train_y, test_y = load_dataset()

# Loading the test data and thus converting them into a test_loader
x = torch.tensor(test_x)
y = torch.tensor(test_y)
np.random.seed(42)
p = np.random.permutation(len(x))
x = x[p]
y= y[p]
dat = TensorDataset(x, y)
test_loader = torch.utils.data.DataLoader(dat,  batch_size=batch_size, shuffle=True)

############################################
#### Initializing models and optimizer  ####
############################################

#### global model ##########
global_model =  SFMNet(n_features= 41, n_classes= 23).cuda()

############## client models ##############
client_gradients = np.zeros([num_clients, 23, 41])   # dimensions of the last layer
client_models = [SFMNet(n_features= 41, n_classes= 23).cuda() for _ in range(num_clients)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model

cs_mat = np.zeros((num_clients, num_clients), dtype=float) * 1e-6

r = [1 / num_clients for i in range(num_clients)]  # trust scores

############### optimizers ################
opt = [optim.SGD(model.parameters(), lr=0.001) for model in client_models]

###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []

clients = np.arange(num_clients)
np.random.seed(42)
dishonest_client_idx =np.random.permutation(num_clients)[:int(0.9 * num_selected)]

# Runnining FL
attack = 0
for round in range(num_rounds):
    probs = [r[i] / sum(r) for i in range(num_clients)]
    np.random.seed(round)
    client_idx = np.random.choice(clients, size=num_selected, replace=False, p=probs)
    # client update
    loss = 0
    print(round)
    for j, i in tqdm(enumerate(client_idx)):
        # get the global weights
        client_models[i].load_state_dict(global_model.state_dict())
        x = torch.tensor(train_x[int(i* len(train_x)/num_clients):int((i+1)*len(train_x)/num_clients)])
        y = torch.tensor(train_y[int(i* len(train_x)/num_clients):int((i+1)*len(train_x)/num_clients)])
        dat = TensorDataset(x, y)
        train_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)
        # honst or not honest update
        if j in dishonest_client_idx:
            attack +=1
            loss += client_update(client_models[i], opt[i], train_loader, epoch=epochs, honest=False)
        else:
            loss += client_update(client_models[i], opt[i], train_loader, epoch=epochs)

        # historical gradient aggregate.
        # historical gradient aggregate.
        client_gradients = weight_aggregate(global_model, client_models, client_gradients, client_idx)
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
        server_aggregate(global_model, client_models, phi, client_idx)
    #print(dishonest_client_idx)
    #print(phi)

    #print(phi)

    attack = 0
    test_loss, acc = test(global_model, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % round)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))

    asr = attack/(len(test_loader)*batch_size)
    print('attack success rate %0.3g' %(asr))
