import model
import utils
import torch
import score_pb2
import verifier

def main():
    print("Models Updates Verifier")

    net = model.Net()
    
    sample = net.state_dict()
    sample = torch.tensor([1,2,3])
    content_bytes = utils.to_bytes(sample)

    modelID = "CNN1"
    parents = ["HckSwavfZ5gceB58aMCYd6wc9Qd5VE2cRhqujiXBfVRv", "CkmuFbBXLgu11PXySQSmdeyodS13TQj67Zse5M6cuDTh"]
    endpoint = "peer1.org1.example.com"
    model_update_pb = utils.to_protobuf(
        modelID=modelID,
        parents=parents,
        content=content_bytes,
        endpoint=endpoint
    )

    content = utils.from_bytes(model_update_pb.content)
    print(content)

    messageID = utils.send_model_update(model_update_pb)
    print(messageID)

    model_update = utils.get_model_update(messageID="FYZWEW8wUDzb66E1jp78iLCYFd8KycVjzkiP9hp5VKkG")
    
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