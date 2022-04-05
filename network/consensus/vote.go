package main

import (
	vpb "github.com/Lekssays/ProxDAG/network/consensus/proto/vote"
	"github.com/Lekssays/ProxDAG/network/plugins/vote"
	"github.com/golang/protobuf/proto"
	"github.com/iotaledger/goshimmer/client"
	"github.com/iotaledger/hive.go/marshalutil"
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
	var decision string
	if votePayload.Decision {
		decision = "true"
	} else {
		decision = "false"
	}
	payload := vote.NewPayload(votePayload.ModelID, votePayload.VoteID, decision, votePayload.Metadata)
	messageID, err := goshimAPI.SendPayload(payload.Bytes())
	if err != nil {
		return "", err
	}
	return messageID, nil
}

func GetVote(messageID string) (vpb.Vote, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	messageRaw, _ := goshimAPI.GetMessage(messageID)
	marshalUtil := marshalutil.New(len(messageRaw.Payload))
	votePayload, err := vote.Parse(marshalUtil.WriteBytes(messageRaw.Payload))
	if err != nil {
		return vpb.Vote{}, err
	}

	var decision bool
	if votePayload.Decision == "true" {
		decision = true
	} else {
		decision = false
	}

	vote := vpb.Vote{
		ModelID:  votePayload.ModelID,
		VoteID:   votePayload.VoteID,
		Decision: decision,
		Metadata: votePayload.Metadata,
	}

	return vote, nil
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
