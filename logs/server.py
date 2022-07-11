import asyncio
import websockets

def write(message: str, filename: str):
    f = open(filename, "a")
    f.write(message + "\n")
    f.close()

async def hello(websocket, path):
    message = await websocket.recv()
    print(message)
    if "!" in message:
        message = message.split("!")
        filename = message[1]
        message = message[0].split("]")
        message = message[1][1:]
        write(message=message, filename=filename)
    else:
        write(message=message, filename="system.log")

start_server = websockets.serve(hello, "0.0.0.0", 7777)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()