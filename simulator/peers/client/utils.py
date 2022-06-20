import base64
import modelUpdate_pb2
import score_pb2
import torch
import io
import requests
import plyvel

from collections import OrderedDict
from google.protobuf import text_format
from io import BytesIO


MODEL_UPDATE_PURPOSE_ID = 17
TRUST_PURPOSE_ID = 21
SIMILARITY_PURPOSE_ID = 22


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
    PROXDAG_ENDPOINT = "http://0.0.0.0:8080/proxdag"

    data = text_format.MessageToString(model_update)

    print(data)
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
    URL = "http://0.0.0.0:5001/api/v0/add"
    files = {'file': BytesIO(content)}
    res = requests.post(URL, files=files)
    return res.json()['Hash']


def get_content_to_ipfs(path: str):
    URL = "http://0.0.0.0:5001/api/v0/get?arg=" + path
    res = requests.post(URL)
    buff = bytes(res.content)
    return buff[512:]


def get_resource_from_leveldb(key: str):
    db = plyvel.DB('./../../../proxdagDB', create_if_missing=False)
    return db.get(bytes(key, encoding='utf8'))


def get_model_update(messageID: str) -> modelUpdate_pb2.ModelUpdate:
    model_update_bytes = get_resource_from_leveldb(key=messageID)
    model_update = None
    if model_update_bytes is None:
        model_update = parse_payload(messageID=messageID)
    else:
        model_update = modelUpdate_pb2.ModelUpdate()
        model_update.ParseFromString(model_update_bytes)
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


def parse_payload(messageID: str):
    GOSHIMMER_ENDPOINT = "http://0.0.0.0:8081/api/message/"
    url = GOSHIMMER_ENDPOINT + messageID
    res = requests.get(url)
    payload = base64.b64decode(res.json()['payload']['content'])
    purpose = str(payload[3:5])
    payload = payload[12:]
    purpose = int("0x" + purpose[4:6] + purpose[8:10], 16)
    parsed_payload = None
    
    if purpose == MODEL_UPDATE_PURPOSE_ID:
        parsed_payload = parse_model_update(payload=payload)
    elif purpose == TRUST_PURPOSE_ID:
        parsed_payload = parse_trust(payload=payload)
    elif purpose == SIMILARITY_PURPOSE_ID:
        parsed_payload = parse_similarity(payload=payload)
    
    return parsed_payload, purpose
