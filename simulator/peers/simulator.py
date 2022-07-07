import asyncio
import json
import subprocess
import time
import os
import argparse
import random
import string
import websockets
import redis
import math

from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--remove',
                        dest = "remove",
                        help = "Remove all ProxDAG containers",
                        default = "false",
                        required = False)
    parser.add_argument('-d', '--dataset',
                        dest = "dataset",
                        help = "Dataset: MNIST, CIFAR, KDD",
                        default = "MNIST",
                        required = False)
    parser.add_argument('-i', '--iterations',
                        dest = "iterations",
                        help = "Number of Iterations",
                        default = 1,
                        required = False)
    parser.add_argument('-p', '--peers',
                        dest = "peers",
                        help = "peers",
                        default = 100,
                        required = False)
    parser.add_argument('-al', '--alpha',
                        dest = "alpha",
                        help = "Dirichlet distribution factor",
                        default = 100,
                        required = False)
    parser.add_argument('-at', '--attack_type',
                        dest = "attack_type",
                        help = "Attack type: lf , backdoor, untargeted",
                        default = "",
                        required = False)
    parser.add_argument('-ap', '--attack_percentage',
                        dest = "attack_percentage",
                        help = "Attack percentage",
                        default = "0",
                        required = False)
    parser.add_argument('-dc', '--dynamic_committee',
                        dest = "dc",
                        help = "Enable dynamic committee",
                        default = "true",
                        required = False)
    return parser.parse_args()


def load_peers():
    peers = []
    with open('peers.json', "r") as f:
        file = json.load(f)
    
    for peer in file['peers']:
        peers.append(peer['name'])

    return peers


def stop_containers(peers, peers_len=100):
    if peers_len > len(peers):
        peers_len = len(peers)
    peers_to_run = " ".join(peers[:peers_len])
    command = "docker stop " + peers_to_run
    subprocess.call(command, shell=True)
    command = "docker rm " + peers_to_run
    subprocess.call(command, shell=True)


def start_containers(peers, peers_len=100):
    if peers_len > len(peers):
        peers_len = len(peers)
    peers_to_run = " ".join(peers[:peers_len])
    command = "docker-compose up -d " + peers_to_run
    subprocess.call(command, shell=True)


def start_learning(dataset, peers, peers_len=100, alpha="0.05", attack_type=None):
    if peers_len > len(peers):
        peers_len = len(peers)

    for peer in peers[:peers_len]:
        if attack_type is None:
            command = "docker exec -it {} python3 /client/main.py -d {} -al {}".format(peer, dataset, alpha)   
        else:
            command = "docker exec -it {} python3 /client/main.py -d {} -al {} -at {}".format(peer, dataset, alpha, attack_type)
        print("Learning ", peer)
        subprocess.call(command, shell=True)


def initialize_protocol():
    command = "cd " + os.getenv("PROTOCOL_PATH") +  " && ./protocol init"
    subprocess.call(command, shell=True)


def run_consensus():
    command = "cd " + os.getenv("PROTOCOL_PATH") +  " && ./protocol consensus"
    subprocess.call(command, shell=True)


def write(filename: str, content: str):
    writing_file = open(filename, "w")
    writing_file.write(content)
    writing_file.close()


def generate_peers_configs(peers: list, num_peers: int, dishonest_peers: list) -> list:
    configs = []
    base_filename = "./templates/peer.yaml"
    for i in range (0, num_peers):
        config_file = open(base_filename, "r")
        content = config_file.read()
        content = content.replace("peer_name", peers[i]['name'])
        content = content.replace("peer_id", peers[i]['id'])
        content = content.replace("my_pub_key", peers[i]['pubkey'])
        if len(dishonest_peers) > 0:
            content = content.replace("dishonest_peers", ",".join(dishonest_peers))
        config_file.close()
        configs.append(content)
    return configs


def generate_docker_compose(configs: list):
    main_config = ""
    base_file = open("./templates/base.yaml", "r")
    base = base_file.read()
    base_file.close()
    main_config = base + "\n"
    for config in configs:
        main_config += config + "\n"
    write(filename="docker-compose.yaml", content=main_config)


def generate_peers(num_peers: int):
    peers = []
    for p in range(0,num_peers):
        tmp = {
            'pubkey': ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase +  string.digits) for _ in range(25)),
            'id': str(p),
            'name': "peer" + str(p) + ".proxdag.io",
        }
        peers.append(tmp)
    
    f = open("peers.json", "w")

    peers_json = {
        "peers": peers,
    }
    f.write(json.dumps(peers_json))
    f.close()

    return peers


def copy_peers():
    print("Copying peers to protocol")
    command = "cp peers.json ./../../protocol/consensus/"
    subprocess.call(command, shell=True)
    command = "cp peers.json ./client/"
    subprocess.call(command, shell=True)


async def send_log(message: str):
    uri = "ws://0.0.0.0:7777"
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    message = dt + " - [" + os.getenv("MY_NAME") + "] " + message
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)

def clear_host_state():
    r = redis.Redis()
    r.flushall()


def generate_dishonest_peers(peers_len, percentage):
    dishonest_peers = []
    if percentage > 0:
        for i in range(0, peers_len):
            dishonest_peers.append(str(i))
        random.shuffle(dishonest_peers)
        limit = math.ceil((percentage/100.0) * peers_len)
        return dishonest_peers[:limit]
    return dishonest_peers


def main():
    print("Simulator :)")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    remove = parse_args().remove
    alpha = parse_args().alpha
    peers_len = int(parse_args().peers)
    dataset = parse_args().dataset
    iterations = int(parse_args().iterations)
    attack_type = parse_args().attack_type
    attack_percentage = int(parse_args().attack_percentage)
    dc = parse_args().dc

    print("Generating docker-compose.yaml")
    peers = generate_peers(num_peers=peers_len)
    
    dishonest_peers = []
    if len(attack_type) > 0:
        dishonest_peers = generate_dishonest_peers(peers_len, attack_percentage)
        print("Dishonest Peers", dishonest_peers)

    configs = generate_peers_configs(peers=peers, num_peers=peers_len, dishonest_peers=dishonest_peers)
    generate_docker_compose(configs=configs)
    copy_peers()

    peers = load_peers()

    if remove == "true":
        stop_containers(peers=peers, peers_len=peers_len)
        return

    settings = "\nlog!dataset = {}, peers = {}, alpha = {}, iterations = {}, dync_committee = {}, attack_type = {}\n".format(dataset, str(peers_len), str(alpha), str(iterations), dc, attack_type)
    loop.run_until_complete(send_log(settings))    
    
    start_containers(peers=peers, peers_len=peers_len)
    time.sleep(5)
    initialize_protocol()
    time.sleep(5)
    for i in range(1, iterations + 1):
        print("Iteration #{}".format(str(i)))
        log_message = "log!\niteration #" + str(i) + "#\n"
        loop.run_until_complete(send_log(log_message))
        if len(attack_type) > 0:
            start_learning(peers=peers, dataset=dataset, peers_len=peers_len, alpha=alpha, attack_type=attack_type)
        else:
            start_learning(peers=peers, dataset=dataset, peers_len=peers_len, alpha=alpha)
        if dc == "true":
            print("\n Generating Scores for Iteration #{}".format(str(i)))
            log_message = "\n\log!scores iteration #" + str(i) + "#\n\n\n"
            loop.run_until_complete(send_log(log_message))
            run_consensus()
            time.sleep(5)

    stop_containers(peers=peers, peers_len=peers_len)
    clear_host_state()


if __name__ == "__main__":
    main()