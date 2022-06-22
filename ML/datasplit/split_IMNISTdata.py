import pickle
import numpy as np
import random
from utils.data.load_dataset import load_torch_dataset
from collections import Counter
import os


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


def _random_weighted_select(probs, size):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return  random.choices(classes, weights=probs, k=size)


def assign_data(data, assignments):
    x = []
    y = []
    for a in assignments:
        x+=random.choices(data[a], k=assignments[a])
        y+= [a]* assignments[a]
    return x, y



alphas = [0.05, 1, 10, 100]
n_clients= 100
n_classes = 10
dataset = 'MNIST'

train_dataset, test_dataset = load_torch_dataset(dataset)


# training
data = {}
n_examples_client = int(60000/n_clients)
for i in range(len(train_dataset)): # or i, image in enumerate(dataset)
    image, label = train_dataset[i] # or whatever your dataset returns
    if label not in data:
        data[label] = []
    data[label].append(image)

clients = {}

for alpha in alphas:
    alpha_vector = [alpha] * n_classes

    for i in range(n_clients):
        assigned_labels_probs = np.random.dirichlet(alpha= (alpha_vector), size = 1).squeeze()
        assigned_labels = _random_weighted_select(assigned_labels_probs, n_examples_client)
        x, y = assign_data(data, Counter(assigned_labels))
        c = client(i, x, y)
        c.write_out(dataset, alpha)

    '''
    print(Counter(assigned_labels[0]))
    print(Counter(assigned_labels[1]))
    '''

# testing
data = {}
n_examples_client = int(10000/n_clients)
for i in range(len(test_dataset)):  # or i, image in enumerate(dataset)
    image, label = train_dataset[i]  # or whatever your dataset returns
    if label not in data:
        data[label] = []
    data[label].append(image)

clients = {}
clients
for alpha in alphas:
    alpha_vector = [alpha] * n_classes
    for i in range(n_clients):
        assigned_labels_probs = np.random.dirichlet(alpha=(alpha_vector), size=1).squeeze()
        assigned_labels = _random_weighted_select(assigned_labels_probs, n_examples_client)
        x, y = assign_data(data, Counter(assigned_labels))
        c = client(i, x, y)
        c.write_out(dataset, alpha, mode='test')




