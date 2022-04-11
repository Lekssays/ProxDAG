module github.com/Lekssays/ProxDAG/network/graph

go 1.16

require (
	github.com/Lekssays/ProxDAG/network/plugins/proxdag v0.0.0
	github.com/cespare/xxhash/v2 v2.1.2 // indirect
	github.com/golang/protobuf v1.5.2
	github.com/gorilla/websocket v1.4.2
	github.com/iotaledger/goshimmer v0.8.8
	github.com/iotaledger/hive.go v0.0.0-20220210121915-5c76c0ccc668
	github.com/onsi/gomega v1.16.0 // indirect
	github.com/syndtr/goleveldb v1.0.0
	google.golang.org/protobuf v1.28.0
)

replace (
	github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate => ./proto/modelUpdate
	github.com/Lekssays/ProxDAG/network/plugins/proxdag v0.0.0 => ../plugins/proxdag
)
