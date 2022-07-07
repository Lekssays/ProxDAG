# ProxDAG
A DAG-based Personalized Fully-Decentralized Learning Framework


# Getting Started
## Dependencies
- Install golang v1.18

- Install python v3.8.10 and pip3

- Install docker v20.10.17 and docker-compose v2.6.1

- Clone GoShimmer (IOTA 2.0) v0.9.1 in Home Directory: 

```git clone https://github.com/iotaledger/goshimmer.git```

- Clone ProxDAG in Home Directory: 

```git clone https://github.com/lekssays/ProxDAG.git```

## Run ProxDAG
- Copy ProxDAG plugin into Goshimmer folder:

```cp -R $HOME/ProxDAG/protocol/plugins/proxdag $HOME/goshimmer/plugins/proxdag```

- Integrate ProxDAG plugin in Goshimmer:
    - Add ```"github.com/Lekssays/ProxDAG/protocol/plugins/proxdag"``` to imports in `$HOME/goshimmer/plugins/research.go`
    - Add ```proxdag.Plugin,"``` after 	```chat.Plugin,``` in `$HOME/goshimmer/plugins/research.go`
    - Install ProxDAG plugin locally ```go get github.com/Lekssays/ProxDAG/protocol/plugins/proxdag```

- Run GoShimmer Network: 

```cd $HOME/goshimmer/tools/docker-network && ./run.sh```

- Create proxdag network 

```docker network create -d bridge proxdag```

- Run IPFS node: 

```docker run -d --name ipfs.proxdag.io --network="proxdag" -v $HOME/ProxDAG/simulator/ipfs/export:/export -v $HOME/ProxDAG/simulator/ipfs/data:/data/ipfs -p 4001:4001 -p 4001:4001/udp -p 0.0.0.0:8088:8088 -p 0.0.0.0:5001:5001 ipfs/go-ipfs:latest```

- Install Python Dependecies: 

```cd $HOME/ProxDAG/simulator/peers/ && pip3 install -r requirements.txt```

- Download and Unzip `data.zip` in `$HOME/ProxDAG/simulator/peers`

- Edit Environment Variables in 

```$HOME/ProxDAG/env.example```

- Rename `example.env` to `.env` and execute `source .env`

- Start Log Server: 

```cd $HOME/ProxDAG/logs/ && python3 server.py```

- Install golang dependencies 

```cd $HOME/ProxDAG/protocol/ && go mod tidy && go build```
