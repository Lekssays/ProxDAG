import asyncio
import websockets
import json

GOSHIMMER_WEBSOCKETS_ENDPOINT = "ws://0.0.0.0:8081/ws"

async def hello():
    async with websockets.connect(GOSHIMMER_WEBSOCKETS_ENDPOINT) as websocket:
        message = await websocket.recv()
        print(message)
        # todo(ahmed): parse the payload
        if "\"payload_type\":787" in message:
            message = json.loads(message)
            print(message)

asyncio.get_event_loop().run_until_complete(hello())
asyncio.get_event_loop().run_forever()