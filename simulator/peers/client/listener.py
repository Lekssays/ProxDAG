import asyncio
import websockets
import utils
import threading
import os

from google.protobuf import text_format


GOSHIMMER_WEBSOCKETS_ENDPOINT = os.getenv("GOSHIMMER_WEBSOCKETS_ENDPOINT")

async def hello():
    async with websockets.connect(GOSHIMMER_WEBSOCKETS_ENDPOINT, ping_interval=None) as websocket:
        while True:
            message = await websocket.recv()
            if "\"payload_type\":787" in message:
                t = threading.Thread(target=utils.process_message, args=(message,))
                t.start()


asyncio.get_event_loop().run_until_complete(hello())
asyncio.get_event_loop().run_forever()
