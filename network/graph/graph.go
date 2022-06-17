package main

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"

	mupb "github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate"
	"github.com/Lekssays/ProxDAG/network/plugins/proxdag"
	"github.com/golang/protobuf/proto"
	"github.com/iotaledger/goshimmer/client"
	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/opt"
	"github.com/syndtr/goleveldb/leveldb/util"
)

const (
	GOSHIMMER_NODE                = "http://0.0.0.0:8080"
	GOSHIMMER_WEBSOCKETS_ENDPOINT = "0.0.0.0:8081"
	REDIS_ENDPOINT                = "http://127.0.0.1:6379"
	LEVELDB_ENDPOINT              = "./../proxdagDB"
	MODEL_UPDATE_PURPOSE_ID       = 17
	INF                           = 1e6
)

type Model struct {
	ID      string
	Updates []string
}

type Response struct {
	MessageID string `json:"messageID,omitempty"`
	Error     string `json:"error,omitempty"`
}

type Message struct {
	Purpose uint32 `json:"purpose"`
	Data    []byte `json:"data"`
}

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

func GetModelUpdate(messageID string) (mupb.ModelUpdate, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	messageRaw, _ := goshimAPI.GetMessage(messageID)
	payload := new(proxdag.Payload)
	err := payload.FromBytes(messageRaw.Payload)
	if err != nil {
		return mupb.ModelUpdate{}, err
	}

	if payload.Purpose() == MODEL_UPDATE_PURPOSE_ID {
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

func SaveModelUpdate(messageID string, modelUpdate mupb.ModelUpdate) error {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return err
	}
	defer db.Close()

	modelUpdateBytes, err := proto.Marshal(&modelUpdate)
	if err != nil {
		return err
	}

	ID := fmt.Sprintf("%s!MU!%s", modelUpdate.ModelID, messageID)
	exists, err := db.Has([]byte(ID), &opt.ReadOptions{})
	if err != nil {
		return err
	}

	if !exists {
		err = db.Put([]byte(ID), modelUpdateBytes, nil)
		if err != nil {
			return err
		}
	}

	return nil
}

func RetrieveModelUpdate(modelID string, messageID string) (*mupb.ModelUpdate, error) {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return &mupb.ModelUpdate{}, err
	}
	defer db.Close()

	ID := fmt.Sprintf("%s!MU!%s", modelID, messageID)
	data, err := db.Get([]byte(ID), nil)
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

func SendModelUpdate(mupdate mupb.ModelUpdate) (string, error) {
	url := GOSHIMMER_NODE + "/proxdag"

	payload := Message{
		Purpose: MODEL_UPDATE_PURPOSE_ID,
		Data:    []byte(mupdate.String()),
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(payloadBytes))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)
	message := string(body)

	var response Response
	json.Unmarshal(body, &response)
	if strings.Contains(message, "messageID") {
		err = SaveModelUpdate(response.MessageID, mupdate)
		if err != nil {
			return "", err
		}

		err = StoreClientID(mupdate.Pubkey, mupdate.ModelID)
		if err != nil {
			return "", err
		}
		return response.MessageID, nil
	}

	return "", errors.New(response.Error)
}

func GetModelUpdatesMessageIDs(modelID string) ([]string, error) {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return []string{}, err
	}
	defer db.Close()

	var key string
	var updates []string

	ID := fmt.Sprintf("%s!MU!", modelID)
	iter := db.NewIterator(util.BytesPrefix([]byte(ID)), nil)
	for iter.Next() {
		key = string(iter.Key())
		messageID := strings.Split(string(key[:]), "!")[2]
		updates = append(updates, messageID)
	}
	iter.Release()

	return updates, iter.Error()
}

func GetModelUpdates(modelID string) ([]*mupb.ModelUpdate, error) {
	messageIDs, err := GetModelUpdatesMessageIDs(modelID)
	if err != nil {
		return []*mupb.ModelUpdate{}, err
	}

	var modelUpdates []*mupb.ModelUpdate
	for i := 0; i < len(messageIDs); i++ {
		modelUpdate, err := RetrieveModelUpdate(modelID, messageIDs[i])
		if err != nil {
			return []*mupb.ModelUpdate{}, err
		}
		modelUpdates = append(modelUpdates, modelUpdate)
	}

	return modelUpdates, nil
}

func StoreClientID(pubkey string, modelID string) error {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return err
	}
	defer db.Close()

	clients, err := GetClients(modelID)
	ID := []byte(strconv.Itoa(len(clients)))
	key := fmt.Sprintf("%s!CL!%s", modelID, pubkey)

	exists, err := db.Has([]byte(key), &opt.ReadOptions{})
	if err != nil {
		return err
	}

	if !exists {
		err = db.Put([]byte(key), []byte(ID), nil)
		if err != nil {
			return err
		}
	}

	return nil
}

func GetClients(modelID string) ([]string, error) {
	messageIDs, err := GetModelUpdatesMessageIDs(modelID)

	if err != nil {
		return []string{}, err
	}

	var clients []string
	for i := 0; i < len(messageIDs); i++ {
		modelUpdate, err := RetrieveModelUpdate(modelID, messageIDs[i])
		if err != nil {
			return []string{}, err
		}
		clients = append(clients, modelUpdate.Pubkey)
	}

	return clients, nil
}

func GetClientID(pubkey string, modelID string) (uint32, error) {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return INF, err
	}
	defer db.Close()

	key := fmt.Sprintf("%s!CL!%s", modelID, pubkey)
	data, err := db.Get([]byte(key), nil)
	if err != nil {
		return INF, err
	}

	ID, err := strconv.Atoi(string(data))
	return uint32(ID), nil
}
