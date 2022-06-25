import argparse
import json
import learning
import utils
import torch
import score_pb2
import verifier
import time

from learning import client, learn
from google.protobuf import text_format

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset',
                        dest = "dataset",
                        help = "Dataset: MNIST, CIFAR, KDD",
                        default = "MNIST",
                        required = True)
    parser.add_argument('-c', '--clients',
                        dest = "num_clients",
                        help = "Number of Clients ",
                        default = 100,
                        required = False)
    parser.add_argument('-sc', '--selected_clients',
                        dest = "num_selected",
                        help = "Number of Selected Clients for Aggregation",
                        default = 10,
                        required = False)
    parser.add_argument('-r', '--rounds',
                        dest = "num_rounds",
                        help = "Number of Training Rounds",
                        default = 100,
                        required = False)
    parser.add_argument('-e', '--epochs',
                        dest = "epochs",
                        help = "Number of Training Epochs",
                        default = 5,
                        required = False)
    parser.add_argument('-bz', '--batch_size',
                        dest = "batch_size",
                        help = "Batch Size",
                        default = 50,
                        required = False)
    parser.add_argument('-dt', '--delta',
                        dest = "delta",
                        help = "Delta",
                        default = 0.5,
                        required = False)
    parser.add_argument('-dc', '--decay',
                        dest = "decay",
                        help = "Decay",
                        default = 0.001,
                        required = False)
    parser.add_argument('-tr', '--threshold',
                        dest = "threshold",
                        help = "Threshold",
                        default = 0.1,
                        required = False)
    parser.add_argument('-al', '--alpha',
                        dest = "alpha",
                        help = "Dirichlet distribution factor",
                        default = 100,
                        required = False)
    parser.add_argument('-at', '--attack_type',
                        dest = "attack_type",
                        help = "Attack type: lf , backdoor, untargeted",
                        default = "lf",
                        required = False)
    return parser.parse_args()


def generate_config():
    dataset = parse_args().dataset
    num_clients = int(parse_args().num_clients)
    num_selected = int(parse_args().num_selected)
    num_rounds = int(parse_args().num_rounds)
    epochs = int(parse_args().epochs)
    batch_size = int(parse_args().batch_size)
    alpha = str(parse_args().alpha)
    attack_type = str(parse_args().attack_type)

    delta = float(parse_args().delta)
    decay = float(parse_args().decay)
    threshold = float(parse_args().threshold)

    config = {
        'dataset': dataset,
        'num_clients': num_clients,
        'num_selected': num_selected,
        'num_rounds': num_rounds,
        'epochs': epochs,
        'batch_size': batch_size,
        'alpha': alpha,
        'delta': delta,
        'decay': decay,
        'threshold': threshold,
        'attack_type': attack_type,
    }

    f = open('config.json', 'w')
    f.write(json.dumps(config))
    f.close()


def main():
    print("Learn :)")
    generate_config()

    weights = torch.tensor([1,2,3])
    gradients = torch.tensor([5,6,7,102])
    weights_bytes = utils.to_bytes(weights)
    weights_path = utils.add_content_to_ipfs(content=weights_bytes)
    accuracy = 0.911

    modelID = "CNN1"
    parents = ["HckSwavfZ5gceB58aMCYd6wc9Qd5VE2cRhqujiXBfVRv", "CkmuFbBXLgu11PXySQSmdeyodS13TQj67Zse5M6cuDTh"]

    messageID = utils.publish_model_update(
        modelID=modelID,
        parents=parents,
        weights=weights_path,
        accuracy=accuracy,
    )

    time.sleep(1)    
    model_update = utils.get_model_update(messageID=messageID)
    print("model_update", model_update)

    learning.learn(modelID=modelID)


if __name__ == "__main__":
    main()
