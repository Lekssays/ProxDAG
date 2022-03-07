package main

import (
	"fmt"

	modelUpdatepb "github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate"
	"github.com/Lekssays/ProxDAG/network/proxdag"
	"github.com/heimdalr/dag"
	"github.com/iotaledger/goshimmer/client"
)

const (
	GOSHIMMERNODE = "http://0.0.0.0:8080"
)

type foobar struct {
	a string
	b string
}

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
	messageRaw, err := goshimAPI.GetMessage(messageID)
	if err != nil {
		return modelUpdatepb.ModelUpdate{}, err
	}
	fmt.Println(messageRaw.Payload)
	parsedPayload, err := ParsePayload(messageRaw.Payload)
	if err != nil {
		return modelUpdatepb.ModelUpdate{}, err
	}
	return parsedPayload, nil
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
// 	panic("todo :)")
// }

// func SaveDAGSnapshot(modelID string, graph dag.DAG) bool {
// 	panic("todo :)")
// }

// func RetrieveDAGSnapshot(hash string) dag.DAG {
// 	panic("todo :)")
// }

func main() {
	fmt.Println("Hello")

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

	graph := dag.NewDAG()

	// response, err := AddModelUpdateEdge(messageID, &graph)
	// if err != nil {
	// 	fmt.Errorf(err.Error())
	// }
	// if response {
	// 	fmt.Println("Vertex and Edges added successfully")
	// }

	// init three vertices
	v1, _ := graph.AddVertex(1)
	v2, _ := graph.AddVertex(2)
	v3, _ := graph.AddVertex(foobar{a: "foo", b: "bar"})

	// add the above vertices and connect them with two edges
	_ = graph.AddEdge(v1, v2)
	_ = graph.AddEdge(v1, v3)

	// describe the graph
	fmt.Print(graph.String())
}
