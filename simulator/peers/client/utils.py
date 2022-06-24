import base64
import modelUpdate_pb2
import score_pb2
import torch
import io
import requests
import plyvel
import json
import os

from collections import OrderedDict, defaultdict
from google.protobuf import text_format
from io import BytesIO


GOSHIMMER_API_ENDPOINT = os.getenv("GOSHIMMER_API_ENDPOINT") # http://0.0.0.0:8081
IPFS_API_ENDPOINT = os.getenv("IPFS_API_ENDPOINT") # http://0.0.0.0:5001
LEVEL_DB_PATH = os.getenv("LEVEL_DB_PATH") # "./../../../proxdagDB"
PROXDAG_ENDPOINT = os.getenv("PROXDAG_ENDPOINT") # "http://0.0.0.0:8080/proxdag"
MY_PUB_KEY = os.getenv("MY_PUB_KEY")


MODEL_UPDATE_PURPOSE_ID = 17
TRUST_PURPOSE_ID = 21
SIMILARITY_PURPOSE_ID = 22


# Chosen Weights to Train From
K = 5

def to_protobuf(modelID: str, parents: list, weights: str, gradients: str, pubkey: str, timestamp: int):
    model_update = modelUpdate_pb2.ModelUpdate()
    model_update.modelID = modelID
    for parent in parents:
        model_update.parents.append(parent)
    model_update.weights = weights
    model_update.gradients = gradients
    model_update.pubkey = pubkey
    model_update.timestamp = timestamp
    return model_update


def to_bytes(content: OrderedDict) -> bytes:
    buff = io.BytesIO()
    torch.save(content, buff)
    buff.seek(0)
    return buff.read()


def from_bytes(content: bytes) -> torch.Tensor:
    buff = io.BytesIO(content)
    loaded_content = torch.load(buff)
    return loaded_content


def send_model_update(model_update: modelUpdate_pb2.ModelUpdate):
    data = text_format.MessageToString(model_update)

    payload = {
        'purpose': 17,
        'data': data,
    }

    res = requests.post(PROXDAG_ENDPOINT, json=payload)

    if "error" not in res.json():
        return res.json()['messageID']  
    return None


def parse_model_update(payload: bytes):
    return text_format.Parse(payload, modelUpdate_pb2.ModelUpdate())


def parse_trust(payload: bytes):
    return text_format.Parse(payload, score_pb2.Trust())


def parse_similarity(payload: bytes):
    return text_format.Parse(payload, score_pb2.Similarity())


def add_content_to_ipfs(content: bytes) -> str:
    url = IPFS_API_ENDPOINT + "/api/v0/add"
    files = {'file': BytesIO(content)}
    res = requests.post(url, files=files)
    return res.json()['Hash']


def get_content_from_ipfs(path: str):
    url = IPFS_API_ENDPOINT + "/api/v0/get?arg=" + path
    res = requests.post(url)
    buff = bytes(res.content)
    return buff[512:]


def get_resource_from_leveldb(key: str):
    db = plyvel.DB(LEVEL_DB_PATH, create_if_missing=True)
    return db.get(bytes(key, encoding='utf8'))


def store_resource_on_leveldb(key: str, content: bytes):
    db = plyvel.DB(LEVEL_DB_PATH, create_if_missing=True)
    db.put(bytes(key, encoding='utf8'), content)


def get_model_update(messageID: str) -> modelUpdate_pb2.ModelUpdate:
    model_update_bytes = get_resource_from_leveldb(key=messageID)
    model_update = None
    if model_update_bytes is None:
        print("not local")
        model_update, _ = parse_payload(messageID=messageID)
    else:
        model_update = parse_model_update(model_update_bytes)

    return model_update


def get_similarity() -> score_pb2.Similarity:
    similarity_bytes = get_resource_from_leveldb(key="similarity")
    if similarity_bytes is None:
        return None
    similarity = score_pb2.Similarity()
    similarity.ParseFromString(similarity_bytes)
    return similarity


def get_trust() -> score_pb2.Trust:
    trust_bytes = get_resource_from_leveldb(key="trust")
    if trust_bytes is None:
        return None
    trust = score_pb2.Trust()
    trust.ParseFromString(trust_bytes)
    return trust


def get_purpose(payload: str):
    payload = base64.b64decode(payload)
    purpose = str(payload[3:5])
    return int("0x" + purpose[4:6] + purpose[8:10], 16)    


def parse_payload(messageID: str):
    url = GOSHIMMER_API_ENDPOINT + "/api/message/" + messageID
    res = requests.get(url)
    payload = base64.b64decode(res.json()['payload']['content'])
    payload = payload[12:]
    purpose = get_purpose(res.json()['payload']['content'])
    parsed_payload = None

    if purpose == MODEL_UPDATE_PURPOSE_ID:
        parsed_payload = parse_model_update(payload=payload)
    elif purpose == TRUST_PURPOSE_ID:
        parsed_payload = parse_trust(payload=payload)
    elif purpose == SIMILARITY_PURPOSE_ID:
        parsed_payload = parse_similarity(payload=payload)
    
    return parsed_payload, purpose


def get_weights(path: str) -> torch.Tensor:
    weights_from_ipfs = get_content_from_ipfs(path=path)
    return from_bytes(weights_from_ipfs)


def get_gradients(path: str) -> torch.Tensor:
    gradients_from_ipfs = get_content_from_ipfs(path=path)
    return from_bytes(gradients_from_ipfs)


def get_weights_to_train(modelID: str, weights_ids: defaultdict):
    weights = []
    chosen_weights =  weights_ids[modelID][0:K]

    trust_score = get_trust()
    similarity = get_similarity()

    metrics = []
    for messageID in chosen_weights:
        mu = get_model_update(messageID=m['messageID'])
        tmp = {
            'trust_score': trust_score[mu.pubkey],
            'similarity': similarity[get_client_id(pubkey=mu.pubkey)][MY_PUB_KEY],
            'messageID': messageID
        }
        metrics.append(tmp)
    
    metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)

    for m in metrics:
        mu = get_model_update(messageID=m['messageID'])
        w = get_weights(path=mu.weights)
        weights.append(w)

    return weights


def get_client_id(pubkey: str):
    with open('clients.json', "r") as f:
        clients = json.load(f)
        return clients[pubkey]
