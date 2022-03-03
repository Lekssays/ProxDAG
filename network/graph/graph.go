package graph

import (
	messagepb "github.com/Lekssays/ProxDAG/network/graph/proto/message"
	"github.com/Lekssays/ProxDAG/network/proxdag"
	"github.com/iotaledger/goshimmer/client"
)

type foobar struct {
	a string
	b string
}

func SendMessage(message messagepb.Message) (string, error) {
	goshimAPI := client.NewGoShimmerAPI("http://0.0.0.0:8080")
	payload := proxdag.NewPayload(message.modelID, message.parentA, message.parentB, message.content)
	messageID, err := goshimAPI.SendPayload(payload.Bytes())
	if err != nil {
		return "", err
	}
	return messageID, nil
}

// func GetMessage(messageID string) (messagepb.Message, error) {
// 	panic("todo :)")
//  messageRaw, err := goshimAPI.GetMessage(messageID)
//  fmt.Println(messageRaw.Payload)
// }

// func AddMessageEdge(messageID string) bool {
// 	// message := GetMessage(messageID)
// 	panic("todo :)")
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

// func main() {
// 	fmt.Println("Hello")

// 	// message := messagepb.Message{
// 	// 	ModelID: "some uuid",
// 	// 	ParentA: "Some hash1",
// 	// 	ParentB: "SOme hash2",
// 	// 	Content: "some base64 string",
// 	// }

// 	// fmt.Println(message)

// 	d := dag.NewDAG()

// 	// init three vertices
// 	v1, _ := d.AddVertex(1)
// 	v2, _ := d.AddVertex(2)
// 	v3, _ := d.AddVertex(foobar{a: "foo", b: "bar"})

// 	// add the above vertices and connect them with two edges
// 	_ = d.AddEdge(v1, v2)
// 	_ = d.AddEdge(v1, v3)

// 	// describe the graph
// 	fmt.Print(d.String())
// }
