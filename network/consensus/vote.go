package main

import (
	"errors"
	"strings"

	vpb "github.com/Lekssays/ProxDAG/network/consensus/proto/vote"
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
)

func SendVote(votePayload vpb.Vote) (string, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	votePayloadBytes, err := proto.Marshal(&votePayload)
	if err != nil {
		return "nil", err
	}

	payload := proxdag.NewPayload("VOTE", string(votePayloadBytes))

	messageID, err := goshimAPI.SendPayload(payload.Bytes())
	if err != nil {
		return "nil", err
	}
	return messageID, nil
}

func GetVote(messageID string) (vpb.Vote, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	messageRaw, _ := goshimAPI.GetMessage(messageID)
	payload, _, err := proxdag.FromBytes(messageRaw.Payload)
	if err != nil {
		return vpb.Vote{}, err
	}

	// todo(ahmed): check model purposeID
	if strings.Contains(string(payload.Data), "vote") {
		var vote vpb.Vote
		err = proto.Unmarshal([]byte(payload.Data), &vote)
		if err != nil {
			return vpb.Vote{}, err
		}
		return vote, nil
	}

	return vpb.Vote{}, errors.New("Unknown payload type!")
}

func SaveVote(vote vpb.Vote) error {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return err
	}
	defer db.Close()

	voteBytes, err := proto.Marshal(&vote)
	if err != nil {
		return err
	}

	err = db.Put([]byte(vote.VoteID), voteBytes, nil)
	if err != nil {
		return err
	}

	return nil
}

func RetrieveVote(voteID string) (*vpb.Vote, error) {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return &vpb.Vote{}, err
	}
	defer db.Close()

	data, err := db.Get([]byte(voteID), nil)
	if err != nil {
		return &vpb.Vote{}, err
	}

	vote := &vpb.Vote{}
	err = proto.Unmarshal(data, vote)
	if err != nil {
		return &vpb.Vote{}, err
	}

	return vote, nil
}
