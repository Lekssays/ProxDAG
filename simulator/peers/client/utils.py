import base64
import modelUpdate_pb2
import torch
import io
import requests

from collections import OrderedDict
from google.protobuf import text_format
from io import BytesIO


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


def get_model_update(messageID: str):
    GOSHIMMER_ENDPOINT = "http://0.0.0.0:8081/api/message/"
    url = GOSHIMMER_ENDPOINT + messageID
    res = requests.get(url)
    payload = base64.b64decode(res.json()['payload']['content'])
    payload = payload[12:]
    return  text_format.Parse(payload, modelUpdate_pb2.ModelUpdate())


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
