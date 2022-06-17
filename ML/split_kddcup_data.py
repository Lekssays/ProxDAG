import pickle
import numpy as np
import random
from utils.data.load_dataset import load_dataset
from collections import Counter
import os

class client:
    def __init__(self, client_id, x, y):
        self.client_id = client_id
        self.x = x
        self.y = y

    def write_out(self, dataset, alpha, mode='train'):
        if not os.path.isdir(dataset):
            os.mkdir(dataset)
        if not os.path.isdir(dataset + '/' + str(self.client_id)):
            os.mkdir(dataset + '/' + str(self.client_id))

        with open(dataset + '/' + str(self.client_id) + '/' + mode + '_' + str(alpha) + '_' + '.pickle', 'wb') as f:
            pickle.dump(self, f)


def _random_weighted_select(probs, size):
    classes = [i for i in range(23)]
    return  random.choices(classes, weights=probs, k=size)


def assign_data(data, assignments):
    x = []
    y = []
    for a in assignments:
        if a in data:
            x+=random.choices(data[a], k=assignments[a])
            y+= [a]* assignments[a]
    return x, y



alphas = [0.05, 1, 10, 100]
n_clients= 100
n_classes = 23
dataset = 'kdd_cup'

x_train, x_test, y_train, y_test = load_dataset()


# training
data = {}
n_examples_client = int(len(x_train)/n_clients)
for i in range(len(x_train)): # or i, image in enumerate(dataset)
    image = x_train[i] # or whatever your dataset returns
    label = y_train[i]
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


# testing
data = {}
n_examples_client = int(len(x_test)/n_clients)
for i in range(len(x_test)): # or i, image in enumerate(dataset)
    image = x_test[i] # or whatever your dataset returns
    label = y_test[i]
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
        c.write_out(dataset, alpha, mode='test')




