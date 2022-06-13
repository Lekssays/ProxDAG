import modelUpdate_pb2
import torch
import io
import requests

from collections import OrderedDict


def to_protobuf(modelID: str, parents: list, content: bytes, endpoint: str):
    model_update = modelUpdate_pb2.ModelUpdate()
    model_update.modelID = modelID
    for parent in parents:
        model_update.parents.append(parent)
    model_update.content = content
    model_update.endpoint = endpoint
    return model_update


def to_bytes(content: OrderedDict) -> bytes:
    buff = io.BytesIO()
    torch.save(content, buff)
    buff.seek(0)
    return buff.read()


def from_bytes(content: bytes):
    buff = io.BytesIO(content)
    loaded_content = torch.load(buff)
    return loaded_content


def send_model_update(model_update: modelUpdate_pb2.ModelUpdate):
    PROXDAG_ENDPOINT = "http://0.0.0.0:8080/chat"

    payload = {
        'purpose': 17,
        'data': str(model_update),
    }

    payload = {
        'to': 'test',
        'from': 'tesst',
        'message': "some id"
    }

    res = requests.post(PROXDAG_ENDPOINT, json=payload)

    if "error" not in res.json():
        return res.json()['messageID']  
    return None


def get_model_update(messageID: str):
    print("todo")
