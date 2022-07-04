import asyncio
import websockets
import json
import utils
import threading
import os

from google.protobuf import text_format


GOSHIMMER_WEBSOCKETS_ENDPOINT = os.getenv("GOSHIMMER_WEBSOCKETS_ENDPOINT")

MODEL_UPDATE_PYTHON_PURPOSE_ID = 16
MODEL_UPDATE_GOLANG_PURPOSE_ID = 17
TRUST_PURPOSE_ID        = 21
SIMILARITY_PURPOSE_ID   = 22
ALIGNMENT_PURPOSE_ID    = 23
GRADIENTS_PURPOSE_ID    = 24
PHI_PURPOSE_ID          = 25

allowed_purposes = [MODEL_UPDATE_PYTHON_PURPOSE_ID, MODEL_UPDATE_GOLANG_PURPOSE_ID, TRUST_PURPOSE_ID, SIMILARITY_PURPOSE_ID, GRADIENTS_PURPOSE_ID, PHI_PURPOSE_ID, ALIGNMENT_PURPOSE_ID]

async def hello():
    async with websockets.connect(GOSHIMMER_WEBSOCKETS_ENDPOINT, ping_interval=None) as websocket:
        while True:
            message = await websocket.recv()
            if "\"payload_type\":787" in message:
                message = json.loads(message)
                messageID = message['data']['id']
                payload, purpose = utils.parse_payload(messageID=messageID)
                print(purpose, payload)
                payload_bytes = bytes(text_format.MessageToString(payload), encoding='utf8')
                if int(purpose) in [MODEL_UPDATE_PYTHON_PURPOSE_ID, MODEL_UPDATE_GOLANG_PURPOSE_ID]:
                    t = threading.Thread(target=utils.store_resource_on_leveldb, args=(messageID, payload_bytes,))
                    t.start()
                    utils.store_weight_id(modelID=payload.modelID, messageID=messageID)
                elif int(purpose) in [TRUST_PURPOSE_ID, SIMILARITY_PURPOSE_ID, GRADIENTS_PURPOSE_ID, PHI_PURPOSE_ID, ALIGNMENT_PURPOSE_ID]:
                    if int(purpose) == TRUST_PURPOSE_ID:
                        t = threading.Thread(target=utils.store_resource_on_leveldb, args=("trust", payload_bytes,))
                        t.start()
                    elif int(purpose) == SIMILARITY_PURPOSE_ID:
                        t = threading.Thread(target=utils.store_resource_on_leveldb, args=("similarity", payload_bytes,))
                        t.start()                
                    elif int(purpose) == GRADIENTS_PURPOSE_ID:
                        t = threading.Thread(target=utils.store_resource_on_leveldb, args=("gradients", payload_bytes,))
                        t.start()
                    elif int(purpose) == ALIGNMENT_PURPOSE_ID:
                        t = threading.Thread(target=utils.store_resource_on_leveldb, args=("algnScore", payload_bytes,))
                        t.start()
                    elif int(purpose) == PHI_PURPOSE_ID:
                        t = threading.Thread(target=utils.store_resource_on_leveldb, args=("phi", payload_bytes,))
                        t.start()

asyncio.get_event_loop().run_until_complete(hello())
asyncio.get_event_loop().run_forever()
