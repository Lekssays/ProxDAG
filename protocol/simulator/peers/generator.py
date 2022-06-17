#!/usr/bin/python3
import argparse

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

def generate_peers_configs(peers: int) -> list:
    configs = []
    base_filename = "./templates/peer.yaml"
    for peer in range(0, peers):
        config_file = open(base_filename, "r")
        content = config_file.read()
        content = content.replace("peer_id", "peer" + str(peer) + ".proxdag.io")
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

def get_peers(filename: str) -> list:
    peers = []
    with open(filename, "r") as f:
        content = f.readlines()
        for peer in content:
            peers.append(peer.strip())
    return peers

def main():
    print("docker-compose.yaml Generator for ProxDAG")
    peers = int(parse_args().peers)
    configs = generate_peers_configs(peers=peers)
    generate_docker_compose(configs=configs)

if __name__ == "__main__":
    main()
