module github.com/Lekssays/ProxDAG/network/graph

go 1.16

require (
	github.com/Lekssays/ProxDAG/network/proxdag v0.0.0-20220225162616-d7378533521a
	github.com/go-redis/redis/v8 v8.11.4 // indirect
	github.com/gomodule/redigo v1.8.8 // indirect
	github.com/heimdalr/dag v1.0.1
	github.com/iotaledger/goshimmer v0.8.8
	google.golang.org/protobuf v1.27.1
)

replace (
	github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate => ./proto/modelUpdate
	github.com/Lekssays/ProxDAG/network/proxdag v0.0.0-20220225162616-d7378533521a => ../proxdag
)
