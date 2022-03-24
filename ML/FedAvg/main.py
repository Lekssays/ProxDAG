import torch

from classification_FedAvg import Trainer
from utils.params.param_parser import parameter_parser


def run_classification():
    cls = Trainer()
    cls.train()
    cls.test()


if __name__ == '__main__':
    args = parameter_parser()
    torch.manual_seed(args.seed)

    run_classification()

