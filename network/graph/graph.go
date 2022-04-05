package main

import (
	"bytes"
	"encoding/gob"

	modelUpdatepb "github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate"
	"github.com/Lekssays/ProxDAG/network/proxdag"
	"github.com/iotaledger/goshimmer/client"
	"github.com/iotaledger/hive.go/marshalutil"
	"github.com/syndtr/goleveldb/leveldb"
)

const (
	GOSHIMMER_NODE                = "http://0.0.0.0:8080"
	GOSHIMMER_WEBSOCKETS_ENDPOINT = "0.0.0.0:8081"
	REDIS_ENDPOINT                = "http://127.0.0.1:6379"
	LEVELDB_ENDPOINT              = "./../proxdagDB"
)

type Node struct {
	MessageID string `bson:"messageID"`
}

type Graph struct {
	ModelID string          `bson:"modelID"`
	AdjList map[Node][]Node `bson:"adjList"`
}

func NewGraph(modelID string) *Graph {
	return &Graph{
		ModelID: modelID,
		AdjList: make(map[Node][]Node),
	}
}

func (graph *Graph) AddNode(node Node) {
	graph.AdjList[node] = []Node{}
}

func (graph *Graph) AddEdge(src Node, dst Node) {
	graph.AdjList[src] = append(graph.AdjList[src], dst)
}

func dfs(graph *Graph, sorting *[]Node, visited map[Node]bool, v Node) {
	visited[v] = true
	for i := 0; i < len(graph.AdjList[v]); i++ {
		if !visited[graph.AdjList[v][i]] {
			dfs(graph, sorting, visited, graph.AdjList[v][i])
		}
	}
	*sorting = append(*sorting, v)
}

func (graph *Graph) TopologicalSort() []Node {
	sorting := make([]Node, 0)
	visited := make(map[Node]bool)

	for node := range graph.AdjList {
		if !visited[node] {
			dfs(graph, &sorting, visited, node)
		}
	}
	return reverse(sorting)
}

func reverse(s []Node) []Node {
	tmp := make([]Node, 0)
	for i := len(s) - 1; i >= 0; i-- {
		tmp = append(tmp, s[i])
	}
	return tmp
}

func (graph *Graph) SaveDAGSnapshot() error {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return err
	}
	defer db.Close()

	var graphBytes bytes.Buffer
	gob.NewEncoder(&graphBytes).Encode(graph)
	err = db.Put([]byte(graph.ModelID), graphBytes.Bytes(), nil)
	if err != nil {
		return err
	}

	return nil
}

func RetrieveDAGSnapshot(modelID string) (Graph, error) {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return Graph{}, err
	}
	defer db.Close()

	data, err := db.Get([]byte(modelID), nil)
	if err != nil {
		return Graph{}, err
	}

	bytesReader := bytes.NewReader(data)
	var graph Graph
	gob.NewDecoder(bytesReader).Decode(&graph)

	return graph, nil
}

func SendModelUpdate(mupdate modelUpdatepb.ModelUpdate) (string, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	payload := proxdag.NewPayload(mupdate.ModelID, mupdate.ParentA, mupdate.ParentB, mupdate.Content, mupdate.Endpoint)
	messageID, err := goshimAPI.SendPayload(payload.Bytes())
	if err != nil {
		return "", err
	}
	return messageID, nil
}

func GetModelUpdate(messageID string) (modelUpdatepb.ModelUpdate, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	messageRaw, _ := goshimAPI.GetMessage(messageID)
	marshalUtil := marshalutil.New(len(messageRaw.Payload))
	modelUpdatePayload, err := proxdag.Parse(marshalUtil.WriteBytes(messageRaw.Payload))
	if err != nil {
		return modelUpdatepb.ModelUpdate{}, err
	}

	modelUpdate := modelUpdatepb.ModelUpdate{
		ModelID:  modelUpdatePayload.ModelID,
		ParentA:  modelUpdatePayload.ParentA,
		ParentB:  modelUpdatePayload.ParentB,
		Content:  modelUpdatePayload.Content,
		Endpoint: modelUpdatePayload.Endpoint,
	}

	return modelUpdate, nil
}

func AddModelUpdateEdge(messageID string, graph Graph) (bool, error) {
	mupdate, err := GetModelUpdate(messageID)
	if err != nil {
		return false, err
	}

	graph.AddNode(Node{MessageID: messageID})
	graph.AddEdge(Node{MessageID: mupdate.ParentA}, Node{MessageID: messageID})
	graph.AddEdge(Node{MessageID: mupdate.ParentB}, Node{MessageID: messageID})

	return true, nil
}
