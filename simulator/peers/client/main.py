import argparse
import json
import learning
import utils
import torch
import score_pb2
import verifier
import time

from learning import client

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
                        help = "Attack type: lf , backdoor",
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
    gradients_bytes = utils.to_bytes(gradients)
    weights_path = utils.add_content_to_ipfs(content=weights_bytes)
    gradients_path = utils.add_content_to_ipfs(content=gradients_bytes)

    modelID = "CNN1"
    parents = ["HckSwavfZ5gceB58aMCYd6wc9Qd5VE2cRhqujiXBfVRv", "CkmuFbBXLgu11PXySQSmdeyodS13TQj67Zse5M6cuDTh"]
    pubkey = "SomePubKey"
    model_update_pb = utils.to_protobuf(
        modelID=modelID,
        parents=parents,
        weights=weights_path,
        gradients=gradients_path,
        pubkey=pubkey,
        timestamp=int(time.time())
    )

    messageID = utils.send_model_update(model_update_pb)
    print(messageID)

    weights_from_ipfs = utils.get_content_to_ipfs(path=model_update_pb.weights)
    weights_from_bytes = utils.from_bytes(weights_from_ipfs)
    print(weights_from_bytes)

    gradients_from_ipfs = utils.get_content_to_ipfs(path=model_update_pb.gradients)
    gradients_from_bytes = utils.from_bytes(gradients_from_ipfs)
    print(gradients_from_bytes)

    model_update = utils.get_model_update(messageID="AC7xrBSfwzb3Y2v5qt6u8B3zPiegJAXaAN7yY7tPtEMV")
    
    # Driver for check_trust
    trust = score_pb2.Trust()
    trust.scores["pk3"] = 0.223
    trust.scores["pk1"] = 0.853
    trust.scores["pk2"] = 0.991
    trust.scores["pk4"] = 0.441
    issuers = ["pk4", "pk2"]
    issuers_trust_scores = verifier.check_trust(trust=trust, issuers=issuers)

    # Driver for check_similarity
    similarity = score_pb2.Similarity()
    similarity.n = len(trust.scores)

    for i in range(0, len(trust.scores)):
        score = score_pb2.Score()
        for j in range(0, len(trust.scores)):
            if i == j:
                score.items.append(0.0)
            else:
                score.items.append((i * 10.0) + j)
        similarity.scores.append(score)
    
    print(similarity.scores[2].items[1])
    sorted_similarities = verifier.check_similarity(similarity=similarity, issuers=issuers_trust_scores)
    print(sorted_similarities)


    local_model, client_models, client_gradients, opt, cs_mat, r = learning.initialize()
    loss, attack = learning.run_fl(
        local_model=local_model,
        client_models=client_models,
        opt=opt,
        r=r,
        alpha=learning.get_parameter(param="alpha"),
        attack_type=learning.get_parameter(param="attack_type"),
        cs_mat=cs_mat,
        client_gradients=client_gradients
    )
    learning.evaluate(local_model=local_model, loss=loss, attack=attack)


if __name__ == "__main__":
    main()
