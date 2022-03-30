module github.com/Lekssays/ProxDAG/network/graph

go 1.16

require (
	github.com/Lekssays/ProxDAG/network/proxdag v0.0.0-20220225162616-d7378533521a
	github.com/cespare/xxhash/v2 v2.1.2 // indirect
	github.com/gomodule/redigo v1.8.8
	github.com/iotaledger/goshimmer v0.8.8
	github.com/iotaledger/hive.go v0.0.0-20220210121915-5c76c0ccc668
	github.com/onsi/gomega v1.16.0 // indirect
	google.golang.org/protobuf v1.27.1
)

replace (
	github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate => ./proto/modelUpdate
	github.com/Lekssays/ProxDAG/network/proxdag v0.0.0-20220225162616-d7378533521a => ../proxdag
)
