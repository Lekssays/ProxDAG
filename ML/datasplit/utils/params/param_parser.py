""""Parameter parsing."""
import argparse
import torch


def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it trains on the minst dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="HIN Transformers.")

    parser.add_argument("--dataset",
                        nargs="?",
                        default="mnist")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for PyTorch. Default is 42.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.01,
                        help="Learning rate. Default is 0.01.")

    parser.add_argument("--iterations",
                        type=int,
                        default=10,
                        help="Epochs. Default is 10.")

    parser.add_argument('--workers',
                        type=int,
                        default=20,
                        help='Number of parallel workers. Default is 3.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64)

    parser.add_argument('--test_batch_size',
                        type=int,
                        default=1000)

    parser.add_argument("--log_interval",
                        default=20,
                        help='how many batches to wait before logging training status')

    parser.add_argument("--use_cuda",
                        type=bool,
                        default=True)

    parser.add_argument("--clients_n",
                        type=int,
                        default=100)

    parser.add_argument("--dir_alpha",
                        type=float,
                        help='alpha for dirichlet distribution',
                        default=100)
    return parser.parse_args()

