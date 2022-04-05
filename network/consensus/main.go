package main

import (
	"fmt"

	vpb "github.com/Lekssays/ProxDAG/network/consensus/proto/vote"
)

func main() {
	vote := vpb.Vote{
		ModelID:  "9313eb37-9fbd-47dc-bcbd-76c9cbf4cce4",
		VoteID:   "vote_iazea55ezze",
		Decision: true,
		Metadata: "{'start_timestamp:1555563296, 'end_timestamp':1655889633, 'electionID':'elec_85596dzz', 'signature':'955522sq3d89dsf4'}",
	}
	messageID, err := SendVote(vote)
	if err != nil {
		fmt.Errorf(err.Error())
	}
	fmt.Printf("MessageID: %s\n", messageID)

	votePayload, _ := GetVote(messageID)
	fmt.Println("Vote:", votePayload.String())

	err = SaveVote(vote)
	if err != nil {
		fmt.Errorf(err.Error())
	}

	rvote, err := RetrieveVote(vote.VoteID)
	if err != nil {
		fmt.Errorf(err.Error())
	}

	fmt.Println("Retrieved Vote:", rvote)
}
