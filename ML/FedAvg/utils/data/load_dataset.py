"""
implementing pytorch-dataset that returns one training point at a given index
for parallel training using pytorch-dataloader
"""
from torchvision import datasets, transforms


def load_dataset():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    return train_dataset, test_dataset

