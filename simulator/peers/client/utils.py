import base64
import modelUpdate_pb2
import score_pb2
import torch
import io
import requests
import plyvel
import json
import os
import pickle
import time

import numpy as np

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


# Limit of Weights to Choose to Analyze and Train From
LIMIT_CHOOSE = 5

# Number of Weights to Train From
LIMIT_SELECTED = 2


ALGN_THRESHOLD = 0.1

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


def get_similarity():
    similarity_path = get_resource_from_leveldb(key="similarity")
    
    if similarity_path is None:
        num_clients = get_parameter(param="num_clients")
        similarity = np.zeros((num_clients, num_clients), dtype=float) * 1e-6
        return similarity

    similarity_bytes = get_content_from_ipfs(path=similarity_path)
    similarity = pickle.loads(similarity_bytes)

    return similarity


def get_trust():
    trust_path = get_resource_from_leveldb(key="trust")
    
    if trust_path is None:
        num_clients = get_parameter(param="num_clients")
        trust = np.zeros((num_clients, num_clients), dtype=float) * 1e-6
        return trust

    trust_bytes = get_content_from_ipfs(path=trust_path)
    trust = pickle.loads(trust_bytes)

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


def get_weights_ids(modelID, limit):
    weights = []
    with open(modelID + ".dat", "r") as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            weights.append(line)

    if limit >= len(weights):
        return weights

    return weights[:limit]


def store_weight_id(modelID, messageID):
    f = open(modelID + ".dat", "a")
    f.write(messageID + "\n")
    f.close()


def clear_weights_ids(modelID):
    f = open(modelID + ".dat", "w")
    f.write("")
    f.close()


def get_weights_to_train(modelID: str):
    weights = []
    indices = []
    parents = []

    chosen_weights_ids = get_weights_ids(modelID=modelID, limit=LIMIT_CHOOSE)
    trust_score = get_trust()
    similarity = get_similarity()

    metrics = []
    for messageID in chosen_weights_ids:
        mu = get_model_update(messageID=m['messageID'])
        tmp = {
            'trust_score': trust_score[mu.pubkey],
            'similarity': similarity[get_client_id(pubkey=mu.pubkey)][MY_PUB_KEY],
            'messageID': messageID
        }
        metrics.append(tmp)
    
    metrics = sorted(metrics, key=lambda x: x['similarity'], reverse=True)

    for m in metrics:
        if 1 - m['trust_score'] > ALGN_THRESHOLD:
            metrics.remove(m)
            continue
        mu = get_model_update(messageID=m['messageID'])
        idx = get_client_id(pubkey=mu.pubkey)
        if idx != int(os.getenv("MY_ID")):
            w = get_weights(path=mu.weights)
            weights.append(w)
            indices.append(idx)
            parents.append(m['messageID'])

    clear_weights_ids(modelID=modelID)

    if len(weights) <= LIMIT_SELECTED - 1:
        return weights, indices, parents
    
    return weights[:LIMIT_SELECTED], indices[:LIMIT_SELECTED], parents[:LIMIT_SELECTED]


def get_client_id(pubkey: str):
    with open('clients.json', "r") as f:
        clients = json.load(f)
        return clients[pubkey]


def get_parameter(param: str):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config[param]


def publish_model_update(modelID, weights, accuracy, parents):
    weights_bytes = to_bytes(weights)
    weights_path = add_content_to_ipfs(content=weights_bytes)

    model_update_pb = to_protobuf(
        modelID=modelID,
        parents=parents,
        weights=weights_path,
        pubkey=os.getenv("MY_PUB_KEY"),
        accuracy=accuracy,
        timestamp=int(time.time())
    )

    return send_model_update(model_update_pb)


def get_phi():
    phi_path = get_resource_from_leveldb(key="phi")
    
    if phi_path is None:
        num_clients = get_parameter(param="num_clients")
        phi = np.zeros((num_clients, num_clients), dtype=float) * 1e-6
        return phi

    phi_bytes = get_content_from_ipfs(path=phi_path)
    phi = pickle.loads(phi_bytes)

    return phi
