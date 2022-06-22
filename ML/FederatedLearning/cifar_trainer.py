###############################
##### importing libraries #####
###############################
import numpy as np
from tqdm import tqdm
import pickle
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader
from models import VGG
from numpy.linalg import norm

torch.backends.cudnn.benchmark=True

torch.manual_seed(42)
np.random.seed(42)

delta = 0.5
threshold = 0.1
decay = 0

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


def client_update(client_model, optimizer, train_loader, epoch=5, honest= True, attack_type= 'lf'):
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
                        if t == 1:
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
        w = model.state_dict()['classifier.4.weight'].cpu().numpy()
        global_w = global_model.state_dict()['classifier.4.weight'].cpu().numpy()
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
        r[i] -= decay
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
                if t == 1 and pred[i] == 7:
                    attack += 1

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc

##### Hyperparameters for federated learning #########
num_clients = 100
num_selected = 30
num_rounds = 10
epochs = 3
batch_size = 50

#############################################################
##### Creating desired data distribution among clients  #####
#############################################################

# Image augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Normalizing the test images
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Loading the test iamges and thus converting them into a test_loader
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        ), batch_size=batch_size, shuffle=True)


############################################
#### Initializing models and optimizer  ####
############################################

#### global model ##########
global_model =  VGG('VGG19').cuda()

############## client models ##############
client_gradients = np.zeros([num_clients, 10, 512])   # dimensions of the last layer
client_models = [ VGG('VGG19').cuda() for _ in range(num_clients)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model

cs_mat = np.zeros((num_clients, num_clients), dtype=float) * 1e-6

r = [1 / num_clients for i in range(num_clients)]  # trust scores

############### optimizers ################
opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []
# Runnining FL

clients = np.arange(num_clients)
np.random.seed(42)
dishonest_client_idx =np.random.permutation(num_clients)[:int(0.5 * num_selected)]

attack = 0
for round in range(num_rounds):
    print(round)
    probs = [r[i] / sum(r) for i in range(num_clients)]
    np.random.seed(round)
    client_idx = np.random.choice(clients, size=num_selected, replace=False, p=probs)
    # client update
    loss = 0
    for j, i in tqdm(enumerate(client_idx)):
        # get the global weights
        client_models[i].load_state_dict(global_model.state_dict())
        # read the local data
        train_obj = pickle.load(open("cifar/" + str(i) + "/train_100_.pickle", "rb"))
        x = torch.stack(train_obj.x)
        y = torch.tensor(train_obj.y)
        dat = TensorDataset(x, y)
        train_loader = DataLoader(dat, batch_size=batch_size, shuffle=True)
        if j in dishonest_client_idx:
            attack+=1
            loss += client_update(client_models[i], opt[i], train_loader, epoch=epochs, honest=False)
        else:
            loss += client_update(client_models[i], opt[i], train_loader, epoch=epochs)

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
    # print(client_scores)
    server_aggregate(global_model, client_models, phi, client_idx)

test_loss, acc = test(global_model, test_loader)
losses_test.append(test_loss)
acc_test.append(acc)
print('%d-th round' % round)
print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))

asr = attack/(len(test_loader)*batch_size)
print('attack success rate %0.3g' %(asr))
