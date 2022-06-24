#!/usr/bin/python3
import argparse
import random
import json
import string

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--peers',
                        dest = "peers",
                        help = "Number of peers",
                        default = 2,
                        required = True)
    return parser.parse_args()


def write(filename: str, content: str):
    writing_file = open(filename, "w")
    writing_file.write(content)
    writing_file.close()


def generate_peers_configs(peers: list, num_peers: int) -> list:
    configs = []
    base_filename = "./templates/peer.yaml"
    for i in range (0, num_peers):
        config_file = open(base_filename, "r")
        content = config_file.read()
        content = content.replace("peer_name", peers[i]['name'])
        content = content.replace("peer_id", peers[i]['id'])
        content = content.replace("my_pub_key", peers[i]['pubkey'])
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


def generate_peers():
    peers = []
    for p in range(0,150):
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


def main():
    print("docker-compose.yaml Generator for ProxDAG")
    num_peers = int(parse_args().peers)

    peers = generate_peers()
    configs = generate_peers_configs(peers=peers, num_peers=num_peers)
    generate_docker_compose(configs=configs)


if __name__ == "__main__":
    main()
