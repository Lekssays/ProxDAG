import asyncio
import base64
import modelUpdate_pb2
import score_pb2
import torch
import io
import requests
import plyvel
import json
import os
import time
import random 
import websockets

import numpy as np

from collections import OrderedDict, defaultdict
from datetime import datetime
from google.protobuf import text_format
from io import BytesIO


GOSHIMMER_API_ENDPOINT = os.getenv("GOSHIMMER_API_ENDPOINT") # http://0.0.0.0:8081
IPFS_API_ENDPOINT = os.getenv("IPFS_API_ENDPOINT") # http://0.0.0.0:5001
LEVEL_DB_PATH = os.getenv("LEVEL_DB_PATH") # "./../../../proxdagDB"
PROXDAG_ENDPOINT = os.getenv("PROXDAG_ENDPOINT") # "http://0.0.0.0:8080/proxdag"
MY_PUB_KEY = os.getenv("MY_PUB_KEY")

MODEL_UPDATE_PYTHON_PURPOSE_ID = 16
MODEL_UPDATE_GOLANG_PURPOSE_ID = 17
TRUST_PURPOSE_ID        = 21
SIMILARITY_PURPOSE_ID   = 22
ALIGNMENT_PURPOSE_ID    = 23
GRADIENTS_PURPOSE_ID    = 24
PHI_PURPOSE_ID          = 25

# Limit of Weights to Choose to Analyze and Train From
LIMIT_CHOOSE = 10

# Number of Weights to Train From
LIMIT_SELECTED = 8


def to_protobuf(modelID: str, parents: list, weights: str, model: str, pubkey: str, timestamp: int, accuracy: float):
    model_update = modelUpdate_pb2.ModelUpdate()
    model_update.modelID = modelID
    for parent in parents:
        model_update.parents.append(parent)
    model_update.weights = weights
    model_update.model = model
    model_update.accuracy = accuracy
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
    payload = {
        'purpose': MODEL_UPDATE_PYTHON_PURPOSE_ID,
        'data': text_format.MessageToString(model_update),
    }

    res = requests.post(PROXDAG_ENDPOINT, json=payload)

    if "error" not in res.json():
        return res.json()['messageID']  
    return None


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
    try:
        db = plyvel.DB(LEVEL_DB_PATH, create_if_missing=True)
        resource =  db.get(bytes(key, encoding='utf8'))
        db.close()
        return resource
    except Exception as e:
        print("ERROR" + str(e.decode("utf-8")))
        time.sleep(2)
        db = plyvel.DB(LEVEL_DB_PATH, create_if_missing=True)
        resource =  db.get(bytes(key, encoding='utf8'))
        db.close()
        print("ERROR" + str(e.decode("utf-8")))
        return resource


def store_resource_on_leveldb(key: str, content: bytes):
    try:
        db = plyvel.DB(LEVEL_DB_PATH, create_if_missing=True)
        db.put(bytes(key, encoding='utf8'), content)
        db.close()
    except Exception as e:
        print("ERROR", str(e))
        time.sleep(2)
        db = plyvel.DB(LEVEL_DB_PATH, create_if_missing=True)
        db.put(bytes(key, encoding='utf8'), content)
        db.close()
        print("FIXED", str(e))


def get_model_update(messageID: str) -> modelUpdate_pb2.ModelUpdate:
    model_update_bytes = get_resource_from_leveldb(key=messageID)
    model_update = None
    if model_update_bytes is None:
        model_update, _ = parse_payload(messageID=messageID)
    else:
        model_update = text_format.Parse(model_update_bytes, modelUpdate_pb2.ModelUpdate())
    return model_update


def get_similarity():
    similarity_bytes = get_resource_from_leveldb(key="similarity")
    
    if similarity_bytes is None:
        num_clients = get_parameter(param="num_clients")
        similarity = np.zeros((num_clients, num_clients), dtype=float) * 1e-6
        return similarity

    similarity_path = str(similarity_bytes).split('"')
    similarity_bytes = get_content_from_ipfs(path=similarity_path[1])
    
    return np.load(BytesIO(similarity_bytes))


def get_trust():
    trust_bytes = get_resource_from_leveldb(key="trust")
    
    if trust_bytes is None:
        num_clients = get_parameter(param="num_clients")
        trust = np.zeros((num_clients, num_clients), dtype=float) * 1e-6
        return trust

    trust_path = str(trust_bytes).split('"')
    trust_bytes = get_content_from_ipfs(path=trust_path[1])
    
    return np.load(BytesIO(trust_bytes))


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

    if purpose == MODEL_UPDATE_GOLANG_PURPOSE_ID:
        payload = base64.b64decode(payload)
        parsed_payload = modelUpdate_pb2.ModelUpdate()
        parsed_payload.ParseFromString(payload)
    elif purpose == MODEL_UPDATE_PYTHON_PURPOSE_ID:
        parsed_payload = text_format.Parse(payload, modelUpdate_pb2.ModelUpdate())
    elif purpose in [TRUST_PURPOSE_ID, SIMILARITY_PURPOSE_ID, GRADIENTS_PURPOSE_ID, PHI_PURPOSE_ID, ALIGNMENT_PURPOSE_ID]:
        payload = base64.b64decode(payload)
        parsed_payload = score_pb2.Score()
        parsed_payload.ParseFromString(payload)
    return parsed_payload, purpose


def get_weights(path: str) -> torch.Tensor:
    weights_from_ipfs = get_content_from_ipfs(path=path)
    return from_bytes(weights_from_ipfs)


def get_gradients(path: str) -> torch.Tensor:
    gradients_from_ipfs = get_content_from_ipfs(path=path)
    return from_bytes(gradients_from_ipfs)


def get_weights_ids(modelID, limit):
    weights = []
    with open(os.getenv("TMP_FOLDER") + modelID + ".dat", "r") as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            if line not in weights:
                weights.append(line)

    if limit >= len(weights):
        return weights

    weights.reverse()
    return weights[:limit]


def store_weight_id(modelID, messageID):
    f = open(os.getenv("TMP_FOLDER") + modelID + ".dat", "a")
    f.write(messageID + "\n")
    f.close()


def get_weights_to_train(modelID: str):
    weights = []
    indices = []
    parents = []
    timestamps = []

    chosen_weights_ids = get_weights_ids(modelID=modelID, limit=LIMIT_CHOOSE)
    trust_score = get_trust()
    similarity = get_similarity()

    metrics = []
    for messageID in chosen_weights_ids:
        mu = get_model_update(messageID=messageID)
        tmp = {
            'trust_score': trust_score[get_client_id(pubkey=mu.pubkey)],
            'similarity': similarity[get_client_id(pubkey=mu.pubkey)][get_client_id(pubkey=MY_PUB_KEY)],
            'messageID': messageID,
            'timestamp': mu.timestamp,
        }
        metrics.append(tmp)
    
    metrics = sorted(metrics, key=lambda x: (x['timestamp'], x['trust_score'], x['similarity']), reverse=True)

    for m in metrics:
        mu = get_model_update(messageID=m['messageID'])
        idx = get_client_id(pubkey=mu.pubkey)
        if idx != int(os.getenv("MY_ID")):
            w = get_weights(path=mu.model)
            if len(w) == 46:
                w = get_weights(path=w)
            weights.append(w)
            indices.append(idx)
            parents.append(m['messageID'])
            timestamps.append(m['timestamp'])

    if len(weights) <= LIMIT_SELECTED:
        return weights, indices, parents

    c = list(zip(weights, indices, parents, timestamps))
    random.shuffle(c)
    weights, indices, parents, timestamps = zip(*c)

    return weights[:LIMIT_SELECTED], indices[:LIMIT_SELECTED], parents[:LIMIT_SELECTED]


def get_client_id(pubkey: str):
    with open('peers.json', "r") as f:
        peers = json.load(f)
    
    for peer in peers['peers']:
        if peer["pubkey"] == pubkey:
            return int(peer["id"])

    return None


def get_parameter(param: str):
    with open("config.json", "r") as f:
        config = json.load(f)
    return config[param]


def publish_model_update(modelID, accuracy, parents, model, weights):
    model_bytes = to_bytes(model)
    model_path = add_content_to_ipfs(content=model_bytes)

    weights_bytes = to_bytes(weights)
    weights_path = add_content_to_ipfs(content=weights_bytes)

    model_update_pb = to_protobuf(
        modelID=modelID,
        parents=parents,
        weights=weights_path,
        pubkey=os.getenv("MY_PUB_KEY"),
        accuracy=accuracy,
        timestamp=int(time.time()),
        model=model_path
    )

    return send_model_update(model_update_pb)


def get_phi():
    phi_path = get_resource_from_leveldb(key="phi")
    
    if phi_path is None:
        num_clients = get_parameter(param="num_clients")
        phi = np.zeros((num_clients, num_clients), dtype=float) * 1e-6
        return phi

    phi_path = str(phi_path).split('"')
    phi_bytes = get_content_from_ipfs(path=phi_path[1])
    return np.load(BytesIO(phi_bytes))


def process_message(message):
    message = json.loads(message)
    messageID = message['data']['id']
    payload, purpose = parse_payload(messageID=messageID)
    print(purpose, payload)
    payload_bytes = bytes(text_format.MessageToString(payload), encoding='utf8')
    if int(purpose) in [MODEL_UPDATE_PYTHON_PURPOSE_ID, MODEL_UPDATE_GOLANG_PURPOSE_ID]:
        store_resource_on_leveldb(messageID, payload_bytes)
        store_weight_id(modelID=payload.modelID, messageID=messageID)
    elif int(purpose) in [TRUST_PURPOSE_ID, SIMILARITY_PURPOSE_ID, GRADIENTS_PURPOSE_ID, PHI_PURPOSE_ID, ALIGNMENT_PURPOSE_ID]:
        if int(purpose) == TRUST_PURPOSE_ID:
            store_resource_on_leveldb("trust", payload_bytes)
        elif int(purpose) == SIMILARITY_PURPOSE_ID:
            store_resource_on_leveldb("similarity", payload_bytes)             
        elif int(purpose) == GRADIENTS_PURPOSE_ID:
            store_resource_on_leveldb("gradients", payload_bytes)
        elif int(purpose) == ALIGNMENT_PURPOSE_ID:
            store_resource_on_leveldb("algnscore", payload_bytes)
        elif int(purpose) == PHI_PURPOSE_ID:
            store_resource_on_leveldb("phi", payload_bytes)


def get_my_latest_accuracy():
    acc_bytes = get_resource_from_leveldb(key='accuracy')
    if acc_bytes == None:
        return 0.00
    return float(str(acc_bytes.decode("utf-8")))


def store_my_latest_accuracy(accuracy: float):
    content = bytes(str(accuracy), encoding="utf-8")
    store_resource_on_leveldb(key="accuracy", content=content)


async def send_log(message: str):
    uri = "ws://172.17.0.1:7777"
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    message = dt + " - [" + os.getenv("MY_NAME") + "] " + message
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)
