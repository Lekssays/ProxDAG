import asyncio
import websockets
import json
import utils
import threading

from collections import defaultdict
from google.protobuf import text_format


GOSHIMMER_WEBSOCKETS_ENDPOINT = "ws://0.0.0.0:8081/ws"


model_updates = defaultdict(list)

async def hello():
    async with websockets.connect(GOSHIMMER_WEBSOCKETS_ENDPOINT, ping_interval=None) as websocket:
        while True:
            message = await websocket.recv()
            if "\"payload_type\":787" in message:
                message = json.loads(message)
                messageID = message['data']['id']
                payload, purpose = utils.parse_payload(messageID=messageID)
                if int(purpose) == utils.MODEL_UPDATE_PURPOSE_ID:
                    payload_bytes = bytes(text_format.MessageToString(payload), encoding='utf8')
                    t = threading.Thread(target=utils.store_resource_on_leveldb, args=(messageID, payload_bytes,))
                    t.start()
                    model_updates[payload.modelID].append(messageID)
                print(model_updates)


asyncio.get_event_loop().run_until_complete(hello())
asyncio.get_event_loop().run_forever()
