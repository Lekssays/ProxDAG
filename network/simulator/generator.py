#!/usr/bin/python3
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--peers',
                        dest = "peers",
                        help = "Peers IP addresses filename",
                        default = "ips.txt",
                        required = True)
    return parser.parse_args()

def write(filename: str, content: str):
    writing_file = open(filename, "w")
    writing_file.write(content)
    writing_file.close()

def generate_peers_configs(peers: list) -> list:
    configs = []
    base_filename = "./templates/peer.yaml"
    for peer in peers:
        config_file = open(base_filename, "r")
        content = config_file.read()
        content = content.replace("core_id", peer + "_peer.proxdag.io")
        content = content.replace("core_ip", peer)
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
    peers_file = parse_args().peers
    peers = get_peers(filename=peers_file)
    configs = generate_peers_configs(peers=peers)
    generate_docker_compose(configs=configs)

if __name__ == "__main__":
    main()
