package main

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"strings"

	mupb "github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate"
	"github.com/Lekssays/ProxDAG/network/plugins/proxdag"
	"github.com/golang/protobuf/proto"
	"github.com/iotaledger/goshimmer/client"
	"github.com/syndtr/goleveldb/leveldb"
)

const (
	GOSHIMMER_NODE                = "http://0.0.0.0:8080"
	GOSHIMMER_WEBSOCKETS_ENDPOINT = "0.0.0.0:8081"
	REDIS_ENDPOINT                = "http://127.0.0.1:6379"
	LEVELDB_ENDPOINT              = "./../proxdagDB"
	MODEL_UPDATE_PURPOSE_ID       = 17
)

type Node struct {
	MessageID string
}

type Graph struct {
	ModelID string
	AdjList map[Node][]Node
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

func SendModelUpdate(mupdate mupb.ModelUpdate) (string, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	mupdatePayloadBytes, err := proto.Marshal(&mupdate)
	if err != nil {
		return "", err
	}

	payload := proxdag.NewPayload(MODEL_UPDATE_PURPOSE_ID, string(mupdatePayloadBytes))
	fmt.Println("Payload:", payload)
	payloadBytes, _ := payload.Bytes()
	messageID, err := goshimAPI.SendPayload(payloadBytes)
	if err != nil {
		return "", err
	}
	return messageID, nil
}

func GetModelUpdate(messageID string) (mupb.ModelUpdate, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	messageRaw, _ := goshimAPI.GetMessage(messageID)
	payload := new(proxdag.Payload)
	err := payload.FromBytes(messageRaw.Payload)
	if err != nil {
		return mupb.ModelUpdate{}, err
	}

	// todo(ahmed): check model purpose
	// if payload.Purpose == MODEL_UPDATE_PURPOSE_ID
	fmt.Println("PayloadAfter:", payload)
	if !strings.Contains(string(payload.Data()), "vote") {
		var mupdate mupb.ModelUpdate
		err = proto.Unmarshal([]byte(payload.Data()), &mupdate)
		if err != nil {
			return mupb.ModelUpdate{}, err
		}
		return mupdate, nil
	}

	return mupb.ModelUpdate{}, errors.New("Unknown payload type!")

}

func AddModelUpdateEdge(messageID string, graph Graph) (bool, error) {
	mupdate, err := GetModelUpdate(messageID)
	if err != nil {
		return false, err
	}

	graph.AddNode(Node{MessageID: messageID})

	for i := 0; i < len(mupdate.Parents); i++ {
		graph.AddEdge(Node{MessageID: mupdate.Parents[i]}, Node{MessageID: messageID})
	}

	return true, nil
}

func SaveModelUpdate(modelUpdate mupb.ModelUpdate) error {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return err
	}
	defer db.Close()

	modelUpdateBytes, err := proto.Marshal(&modelUpdate)
	if err != nil {
		return err
	}

	err = db.Put([]byte(modelUpdate.ModelID), modelUpdateBytes, nil)
	if err != nil {
		return err
	}

	return nil
}

func RetrieveModelUpdate(modelID string) (*mupb.ModelUpdate, error) {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return &mupb.ModelUpdate{}, err
	}
	defer db.Close()

	data, err := db.Get([]byte(modelID), nil)
	if err != nil {
		return &mupb.ModelUpdate{}, err
	}

	modelUpdate := &mupb.ModelUpdate{}
	err = proto.Unmarshal(data, modelUpdate)
	if err != nil {
		return &mupb.ModelUpdate{}, err
	}

	return modelUpdate, nil
}
