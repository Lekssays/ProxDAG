#!/bin/bash

docker network create proxdag

# Run IPFS node
docker run -d --name ipfs.proxdag.io --network="outside" -v /home/ahmed/workspace/ProxDAG/simulator/ipfs/export:/export -v /home/ahmed/workspace/ProxDAG/simulator/ipfs/data:/data/ipfs -p 4001:4001 -p 4001:4001/udp -p 0.0.0.0:8088:8088 -p 0.0.0.0:5001:5001 ipfs/go-ipfs:latest

# Run GoShimmer node
cd ./goshimmer/
docker-compose up -d

sleep 10

# Run ProxDAG nodes
cd ../peers/
docker-compose up -d
