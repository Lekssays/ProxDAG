package main

import (
	"fmt"

	modelUpdatepb "github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate"
	"github.com/Lekssays/ProxDAG/network/proxdag"
	"github.com/iotaledger/goshimmer/client"
	"github.com/iotaledger/hive.go/marshalutil"
)

const (
	GOSHIMMERNODE = "http://0.0.0.0:8080"
)

func SendModelUpdate(mupdate modelUpdatepb.ModelUpdate) (string, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMERNODE)
	payload := proxdag.NewPayload(mupdate.ModelID, mupdate.ParentA, mupdate.ParentB, mupdate.Content)
	fmt.Println("Content-length:", len(mupdate.Content))
	messageID, err := goshimAPI.SendPayload(payload.Bytes())
	if err != nil {
		return "", err
	}
	return messageID, nil
}

func ParsePayload([]byte) (modelUpdatepb.ModelUpdate, error) {
	panic("todo:)")
}

func GetModelUpdate(messageID string) (modelUpdatepb.ModelUpdate, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMERNODE)
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
	}
	return modelUpdate, nil
}

// func AddModelUpdateEdge(messageID string, graph *dag.DAG) (bool, error) {
// 	mupdate, err := GetModelUpdate(messageID)
// 	if err != nil {
// 		return false, err
// 	}

// 	// mupdate := modelUpdatepb.ModelUpdate{
// 	// 	ModelID: "b494b247-777e-4de3-a3f0-b2f7c0107c9d",
// 	// 	ParentA: "3yZ6ZzmsV6tcUN8JWVQJSFRMsqHXXeLqzHpQy1yfkV9w",
// 	// 	ParentB: "Cnt7S5KbAmMsu9azEDksR9xqaD21y6hjE8ENXPpJ5z7Q",
// 	// 	Content: "some base64 string",
// 	// }

// 	mupdateParentA, err := GetModelUpdate(mupdate.ParentA)
// 	if err != nil {
// 		return false, err
// 	}
// 	mupdateParentB, err := GetModelUpdate(mupdate.ParentB)
// 	if err != nil {
// 		return false, err
// 	}

// 	graph.AddVertex(mupdate)
// 	graph.AddEdge(mupdate, mupdateParentA)
// 	graph.AddEdge(mupdate, mupdateParentB)

// 	return true, nil
// }

// func HashDAG(graph dag.DAG) string {
// 		panic("todo :)")
// }

// func SaveDAGSnapshot(modelID string, graph dag.DAG) bool {
// 	panic("todo :)")
// }

// func RetrieveDAGSnapshot(hash string) dag.DAG {
// 	panic("todo :)")
// }

type Node struct {
	MessageID string
}

type Graph struct {
	AdjList map[Node][]Node
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
	return Reverse(sorting)
}

func Reverse(s []Node) []Node {
	tmp := make([]Node, 0)
	for i := len(s) - 1; i >= 0; i-- {
		tmp = append(tmp, s[i])
	}
	return tmp
}

func main() {
	fmt.Println("Hello")

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

	mupdate := modelUpdatepb.ModelUpdate{
		ModelID: "9313eb37-9fbd-47dc-bcbd-76c9cbf4cce4",
		ParentA: "GfnVharJcoV73nT3QiNqm6yXRGkocvw5HoiwwWzu2Dc3",
		ParentB: "5SSTDBDHhstyRavjexGzLWDKxs1bckwkgxeLP9BLpDW9",
		Content: "some",
	}
	messageID, err := SendModelUpdate(mupdate)
	if err != nil {
		fmt.Errorf(err.Error())
	}
	fmt.Printf("MessageID: %s\n", messageID)

	modelUpdate, _ := GetModelUpdate("4LU1ME6XbzELT4g6HMJ9xU115EUJLgvLeeqfEKFD67Zv")
	fmt.Println(modelUpdate.String())
}
