import torch

from classification import Trainer
from utils.params.param_parser import parameter_parser

args = parameter_parser()


def run_classification():
    cls = Trainer(args)
    cls.train()
    cls.test()


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    dataset = 'kddcup'

    if dataset == 'cifar':
        args.dataset = 'cifar'
        args.batch_size = 128
        args.iterations = 100
        args.learning_rate = 0.1

    elif dataset == 'kddcup':
        args.dataset = 'kddcup'

    run_classification()

