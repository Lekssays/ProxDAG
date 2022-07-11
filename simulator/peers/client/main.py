import argparse
import json
import learning
import utils
import torch
import time
import os
import pickle

from os.path import exists


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


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset',
                        dest = "dataset",
                        help = "Dataset: MNIST, CIFAR, KDD",
                        default = "MNIST",
                        required = True)
    parser.add_argument('-p', '--peers',
                        dest = "num_peers",
                        help = "Number of Peers ",
                        default = 100,
                        required = False)
    parser.add_argument('-i', '--iterations',
                        dest = "iterations",
                        help = "Number of Iterations",
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
    parser.add_argument('-ap', '--attack_percentage',
                        dest = "attack_percentage",
                        help = "Attack percentage",
                        default = "0",
                        required = False)
    parser.add_argument('-dc', '--dynamic_committee',
                        dest = "dc",
                        help = "Enable dynamic committee",
                        default = "true",
                        required = False)
    return parser.parse_args()


def generate_config():

    dataset = parse_args().dataset
    num_peers = int(parse_args().num_peers)
    epochs = int(parse_args().epochs)
    batch_size = int(parse_args().batch_size)
    alpha = str(parse_args().alpha)
    attack_type = str(parse_args().attack_type)
    iterations = int(parse_args().iterations)
    dc = parse_args().dc
    attack_percentage = int(parse_args().attack_percentage)

    config = {
        'dataset': dataset,
        'num_peers': num_peers,
        'iterations': iterations,
        'epochs': epochs,
        'batch_size': batch_size,
        'alpha': alpha,
        'attack_type': attack_type,
        'dc': dc,
        'attack_percentage': attack_percentage,
    }

    f = open('config.json', 'w')
    f.write(json.dumps(config))
    f.close()

    return dataset


def main():
    generate_config()

    modelID = "9313eb37-9fbd-47dc-bcbd-76c9cbf4cce4"
    if not exists(os.getenv("TMP_FOLDER") + modelID + ".dat"):
        local_model = learning.initialize(modelID)
        weights_bytes = utils.to_bytes(local_model.state_dict()['fc.weight'].cpu().numpy())
        weights_path = utils.add_content_to_ipfs(content=weights_bytes)
        model_bytes = utils.to_bytes(local_model.state_dict())
        model_path = utils.add_content_to_ipfs(content=model_bytes)
        messageID = utils.publish_model_update(
            modelID=modelID,
            parents=[],
            weights=weights_path,
            model=model_path,
            accuracy=0.0,
        )
        utils.store_my_latest_accuracy(accuracy=0.00)
        utils.store_weight_id(modelID=modelID, messageID=messageID)

    learning.learn(modelID=modelID)


if __name__ == "__main__":
    main()
