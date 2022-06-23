import asyncio
import websockets
import json
import utils

GOSHIMMER_WEBSOCKETS_ENDPOINT = "ws://0.0.0.0:8081/ws"

async def hello():
    async with websockets.connect(GOSHIMMER_WEBSOCKETS_ENDPOINT, ping_interval=None) as websocket:
        while True:
            message = await websocket.recv()
            if "\"payload_type\":787" in message:
                message = json.loads(message)
                payload, _ = utils.parse_payload(messageID=message['data']['id'])
                print(payload)

asyncio.get_event_loop().run_until_complete(hello())
asyncio.get_event_loop().run_forever()
