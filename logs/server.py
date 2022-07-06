import asyncio
import websockets

def write(message: str):
    f = open("system.log", "a")
    f.write(message + "\n")
    f.close()

async def hello(websocket, path):
    message = await websocket.recv()
    print(message)
    write(message=message)

start_server = websockets.serve(hello, "0.0.0.0", 7777)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()