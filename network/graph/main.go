package main

import (
	"fmt"
	"os"
	"sync"
	"time"

	modelUpdatepb "github.com/Lekssays/ProxDAG/network/graph/proto/modelUpdate"
)

func main() {
	args := os.Args[1:]

	if args[0] == "test" {
		graph := NewGraph("modelID1")
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

		err := graph.SaveDAGSnapshot()
		if err != nil {
			fmt.Println(err.Error())
		}
		graphNew, err := RetrieveDAGSnapshot("modelID1")
		if err != nil {
			fmt.Println(err.Error())
		}

		fmt.Println("Saved Graph:", graphNew)

		mupdate := modelUpdatepb.ModelUpdate{
			ModelID:  "9313eb37-9fbd-47dc-bcbd-76c9cbf4cce4",
			ParentA:  "GfnVharJcoV73nT3QiNqm6yXRGkocvw5HoiwwWzu2Dc3",
			ParentB:  "5SSTDBDHhstyRavjexGzLWDKxs1bckwkgxeLP9BLpDW9",
			Content:  "some",
			Endpoint: "peer0.proxdag.io:5696",
		}
		messageID, err := SendModelUpdate(mupdate)
		if err != nil {
			fmt.Errorf(err.Error())
		}
		fmt.Printf("MessageID: %s\n", messageID)

		modelUpdate, _ := GetModelUpdate(messageID)
		fmt.Println(modelUpdate.String())

		AddModelUpdateEdge(messageID, *graph)
		fmt.Println(graph)

		err = SaveModelUpdate(mupdate)
		if err != nil {
			fmt.Errorf(err.Error())
		}

		rmupdate, err := RetrieveModelUpdate(mupdate.ModelID)
		if err != nil {
			fmt.Errorf(err.Error())
		}
		
		fmt.Println("Retrieved ModelUpdate:", rmupdate)
	} else if args[0] == "listener" {
		var wg sync.WaitGroup
		for {
			timer := time.After(6 * time.Second)
			wg.Add(1)
			go RunLiveFeed(&wg)
			wg.Wait()
			<-timer
		}
	}
}
