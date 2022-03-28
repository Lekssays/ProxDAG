package main

import (
	"bytes"
	"context"
	"encoding/gob"
	"fmt"
	"time"

	modelUpdatepb "github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate"
	"github.com/Lekssays/ProxDAG/network/proxdag"
	"github.com/gomodule/redigo/redis"
	"github.com/iotaledger/goshimmer/client"
	"github.com/iotaledger/hive.go/marshalutil"
)

const (
	GOSHIMMER_NODE = "http://0.0.0.0:8080"
	REDIS_ENDPOINT = "http://127.0.0.1:6379"
)

type Node struct {
	MessageID string
}

type Graph struct {
	AdjList map[Node][]Node
}

type Model struct {
	Models map[string][]Graph
}

func NewGraph() *Graph {
	return &Graph{
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

func SaveDAGSnapshot(modelID string, graph Graph) {
	pool := &redis.Pool{
		DialContext: func(ctx context.Context) (redis.Conn, error) {
			return redis.Dial("tcp", REDIS_ENDPOINT)
		},

		MaxIdle:     1024,
		IdleTimeout: 5 * time.Minute,
	}

	conn := pool.Get()
	defer conn.Close()

	var buf bytes.Buffer
	gob.NewEncoder(&buf).Encode(graph)
	conn.Do("SET", modelID, buf.Bytes())
}

func RetrieveDAGSnapshot(modelID string) Graph {
	pool := &redis.Pool{
		DialContext: func(ctx context.Context) (redis.Conn, error) {
			return redis.Dial("tcp", REDIS_ENDPOINT)
		},

		MaxIdle:     1024,
		IdleTimeout: 5 * time.Minute,
	}

	conn := pool.Get()
	defer conn.Close()

	bs, _ := redis.Bytes(conn.Do("GET", modelID))
	bytesReader := bytes.NewReader(bs)

	var graph Graph
	gob.NewDecoder(bytesReader).Decode(&graph)

	return graph
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
	fmt.Println(modelUpdatePayload.Content)
	if err != nil {
		return modelUpdatepb.ModelUpdate{}, err
	}
	modelUpdate := modelUpdatepb.ModelUpdate{
		ModelID: modelUpdatePayload.ModelID,
		ParentA: modelUpdatePayload.ParentA,
		ParentB: modelUpdatePayload.ParentB,
		Content: modelUpdatePayload.Content,
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

func main() {
	graph := NewGraph()
	n1 := Node{
		MessageID: "A",
	}
	n2 := Node{
		MessageID: "B",
	}
	n3 := Node{
		MessageID: "C",
	}
	n4 := Node{
		MessageID: "D",
	}
	graph.AddNode(n1)
	graph.AddNode(n2)
	graph.AddNode(n3)
	graph.AddNode(n4)

	graph.AddEdge(n1, n2)
	graph.AddEdge(n1, n3)
	graph.AddEdge(n2, n4)

	fmt.Println(graph)
	fmt.Println(graph.TopologicalSort())

	SaveDAGSnapshot("test", *graph)
	graphNew := RetrieveDAGSnapshot("test")
	fmt.Println("Saved Graph:", graphNew)

	mupdate := modelUpdatepb.ModelUpdate{
		ModelID: "9313eb37-9fbd-47dc-bcbd-76c9cbf4cce4",
		ParentA: "GfnVharJcoV73nT3QiNqm6yXRGkocvw5HoiwwWzu2Dc3",
		ParentB: "5SSTDBDHhstyRavjexGzLWDKxs1bckwkgxeLP9BLpDW9",
		Content: "some",
		Endpoint: "peer0.proxdag.io:5696",
	}
	messageID, err := SendModelUpdate(mupdate)
	if err != nil {
		fmt.Errorf(err.Error())
	}
	fmt.Printf("MessageID: %s\n", messageID)

	modelUpdate, _ := GetModelUpdate("4LU1ME6XbzELT4g6HMJ9xU115EUJLgvLeeqfEKFD67Zv")
	fmt.Println(modelUpdate.String())

	AddModelUpdateEdge(messageID, *graph)
	fmt.Println(graph)
}
