"""
implementing pytorch-dataset that returns one training point at a given index
for parallel training using pytorch-dataloader
"""
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def load_torch_dataset(dataset='MNIST'):
    if dataset == 'MNIST':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    elif dataset == 'cifar':
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])

        train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('../data', train=False, transform=transform)

    return train_dataset, test_dataset


def load_dataset(dataset='kddcup'):

    if dataset == 'kddcup':
        kdd_df = pd.read_csv('./utils/data/dataset/kddcup.csv', delimiter=',', header=None)
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

        y = kdd_df.iloc[:,-1].values
        x = kdd_df.iloc[:,0:-1].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test
