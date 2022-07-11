import asyncio
import websockets

def write(message: str, filename="system.log"):
    f = open(filename, "a")
    f.write(message + "\n")
    f.close()

async def hello(websocket, path):
    message = await websocket.recv()
    print(message)
    message = message.split("!")
    if len(message) == 2:
        write(message=message[1], filename=message[0])
    else:
        write(message=message[0])

start_server = websockets.serve(hello, "0.0.0.0", 7777)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()