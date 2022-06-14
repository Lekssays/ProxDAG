import modelUpdate_pb2
import utils
import score_pb2

LIMIT_PARENTS = 2
MY_PUB_KEY = "pk1"

def verify_model_update(model_update: str):
    print("todo")


def check_trust(trust: score_pb2.Trust, issuers: list):
    scores = {}

    for k in trust.scores:
        if k in issuers:
            scores[k] = trust.scores[k]

    issuers_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    return issuers_scores


def _get_clients_ids(public_key: str):
    clients = {
        "pk1": 0,
        "pk2": 1,
        "pk3": 2,
        "pk4": 3
    }
    return clients[public_key]


def check_similarity(similarity: score_pb2.Similarity, issuers: dict):
    similarities = {}

    for issuer in issuers:
        if issuer != MY_PUB_KEY:
            similarities[issuer] = similarity.scores[_get_clients_ids(MY_PUB_KEY)].items[_get_clients_ids(issuer)]

    return dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
