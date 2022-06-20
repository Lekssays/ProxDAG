from sqlite3 import Timestamp
import model
import utils
import torch
import score_pb2
import verifier
import time

def main():
    print("Models Updates Verifier")

    net = model.Net()
    
    sample = net.state_dict()
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

if __name__ == '__main__':
    main()