package main

import (
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/Lekssays/ProxDAG/protocol/committee"
	"github.com/Lekssays/ProxDAG/protocol/consensus"
	vpb "github.com/Lekssays/ProxDAG/protocol/consensus/proto/vote"
	"github.com/Lekssays/ProxDAG/protocol/graph"
	mupb "github.com/Lekssays/ProxDAG/protocol/graph/proto/modelUpdate"
)

func main() {
	args := os.Args[1:]

	if len(args) < 1 {
		fmt.Errorf("Please specify which component you want to test :)")
	} else {
		if args[0] == "graph" {
			fmt.Println("Running DAG")
			g := graph.NewGraph("modelID1")
			n1 := graph.Node{
				MessageID: "A",
			}
			n2 := graph.Node{
				MessageID: "B",
			}
			n3 := graph.Node{
				MessageID: "C",
			}
			n4 := graph.Node{
				MessageID: "D",
			}
			g.AddNode(n1)
			g.AddNode(n2)
			g.AddNode(n3)
			g.AddNode(n4)

			g.AddEdge(n1, n2)
			g.AddEdge(n1, n3)
			g.AddEdge(n2, n4)

			fmt.Println(g)
			fmt.Println(g.TopologicalSort())

			err := g.SaveDAGSnapshot()
			if err != nil {
				fmt.Println(err.Error())
			}
			graphNew, err := graph.RetrieveDAGSnapshot("modelID1")
			if err != nil {
				fmt.Println(err.Error())
			}

			fmt.Println("Saved Graph:", graphNew)

			mupdate := mupb.ModelUpdate{
				ModelID:  "9313eb37-9fbd-47dc-bcbd-76c9cbf4cce4",
				Parents:  []string{"GfnVharJcoV73nT3QiNqm6yXRGkocvw5HoiwwWzu2Dc3", "GfnVharJcoV73nT3QiNqm6yXRGkocvw5HoiwwWzu2Dc3", "GfnVharJcoV73nT3QiNqm6yXRGkocvw5HoiwwWzu2Dc3"},
				Hash:  "SomeIPFSHash",
				Endpoint: "peer0.proxdag.io:5696",
				Pubkey:   "pubkey1",
			}

			messageID, err := graph.SendModelUpdate(mupdate)
			if err != nil {
				fmt.Errorf(err.Error())
			}
			fmt.Printf("MessageID: %s\n", messageID)

			modelUpdate, _ := graph.GetModelUpdate(messageID)
			fmt.Println(modelUpdate.String())

			graph.AddModelUpdateEdge(messageID, *g)
			fmt.Println(g)

			err = graph.SaveModelUpdate(messageID, mupdate)
			if err != nil {
				fmt.Errorf(err.Error())
			}

			rmupdate, err := graph.RetrieveModelUpdate(mupdate.ModelID, messageID)
			if err != nil {
				fmt.Errorf(err.Error())
			}

			fmt.Println("Retrieved ModelUpdate:", rmupdate)

			updates, err := graph.GetModelUpdates(mupdate.ModelID)
			if err != nil {
				fmt.Errorf(err.Error())
			}
			fmt.Println("GetModelUpdates", updates)

			clients, err := graph.GetClients(mupdate.ModelID)
			if err != nil {
				fmt.Errorf(err.Error())
			}
			fmt.Println("GetClients", clients)

			clientID, err := graph.GetClientID(clients[0], mupdate.ModelID)
			if err != nil {
				fmt.Errorf(err.Error())
			}
			fmt.Println("GetClientID", clientID)

		} else if args[0] == "listener" {
			fmt.Println("Running Listener")
			var wg sync.WaitGroup
			for {
				timer := time.After(6 * time.Second)
				wg.Add(1)
				go graph.RunLiveFeed(&wg)
				wg.Wait()
				<-timer
			}
		} else if args[0] == "consensus" {
			fmt.Println("Running Consensus")
			vote := vpb.Vote{
				ModelID:  "9313eb37-9fbd-47dc-bcbd-76c9cbf4cce4",
				VoteID:   "vote_iazea55ezze",
				Decision: true,
				Metadata: "{'start_timestamp:1555563296, 'end_timestamp':1655889633, 'electionID':'elec_85596dzz', 'signature':'955522sq3d89dsf4'}",
			}
			messageID, err := consensus.SendVote(vote)
			if err != nil {
				fmt.Errorf(err.Error())
			}
			fmt.Printf("MessageID: %s\n", messageID)

			votePayload, _ := consensus.GetVote(messageID)
			fmt.Println("Vote:", votePayload)

			err = consensus.SaveVote(vote)
			if err != nil {
				fmt.Errorf(err.Error())
			}

			rvote, err := consensus.RetrieveVote(vote.VoteID)
			if err != nil {
				fmt.Errorf(err.Error())
			}

			fmt.Println("Retrieved Vote:", rvote)
		} else if args[0] == "committee" {
			fmt.Println("Running Dynamic Committee")
			_, _, err := committee.GenerateVRFKeys()
			if err != nil {
				fmt.Errorf(err.Error())
			}

			message := []byte("ProxDAG is amazing")
			_, proof, err := committee.Prove(message)
			if err != nil {
				fmt.Errorf(err.Error())
			}

			verified, err := committee.VerifyVRF(message, proof)
			if err != nil {
				fmt.Errorf(err.Error())
			}

			if verified {
				fmt.Println("Output is verified :)")
			} else {
				fmt.Println("Output is NOT verified :)")
			}
		} else {
			fmt.Errorf("Invalid Operation!")
		}
	}
}
