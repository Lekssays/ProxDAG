import json
import subprocess
import time
import random
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--clear',
                        dest = "clear",
                        help = "Remove all ProxDAG containers",
                        default = "false",
                        required = False)
    
    return parser.parse_args()

def load_peers():
    peers = []
    with open('peers.json', "r") as f:
        file = json.load(f)
    
    for peer in file['peers']:
        peers.append(peer['name'])

    return peers


def stop_containers(peers, limit=100):
    if limit > len(peers):
        limit = len(peers)
    peers_to_run = " ".join(peers[:limit])
    command = "docker stop " + peers_to_run
    subprocess.call(command, shell=True)
    command = "docker rm " + peers_to_run
    subprocess.call(command, shell=True)

def start_containers(peers, limit=100):
    if limit > len(peers):
        limit = len(peers)
    peers_to_run = " ".join(peers[:limit])
    command = "docker-compose up -d " + peers_to_run
    subprocess.call(command, shell=True)

def start_learning(dataset, peers, limit=100):
    if limit > len(peers):
        limit = len(peers)

    for peer in peers[:limit]:
        command = "docker exec -it {} python3 /client/main.py -d {}".format(peer, dataset)
        print("Learning ", peer)
        subprocess.call(command, shell=True)
        w = random.randint(1, 10)
        print("Sleeping for {} seconds".format(str(w)))
        time.sleep(w)


def initialize_protocol():
    command = "cd " + os.getenv("PROTOCOL_PATH") +  " && ./protocol init"
    subprocess.call(command, shell=True)



def main():
    print("Simulator")

    clear = parse_args().clear

    peers = load_peers()
    dataset = "MNIST"
    limit = 10

    if clear == "true":
        stop_containers(peers=peers, limit=limit)
        return

    start_containers(peers=peers, limit=limit)
    
    time.sleep(3)

    initialize_protocol()

    start_learning(peers=peers, dataset=dataset, limit=limit)

    

if __name__ == "__main__":
    main()