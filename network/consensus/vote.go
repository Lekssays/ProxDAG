package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/http"
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
	VOTE_PURPOSE_ID               = 18
)

type Message struct {
	Purpose uint32 `json:"purpose"`
	Data    []byte `json:"data"`
}

func SendVote(votePayload vpb.Vote) (string, error) {
	url := GOSHIMMER_NODE + "/proxdag"

	payload := Message{
		Purpose: VOTE_PURPOSE_ID,
		Data:    []byte(votePayload.String()),
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
	if strings.Contains(message, "messageID") {
		return message[14:58], nil
	}

	return "", errors.New(message)
}

func GetVote(messageID string) (vpb.Vote, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	messageRaw, _ := goshimAPI.GetMessage(messageID)
	payload := new(proxdag.Payload)
	err := payload.FromBytes(messageRaw.Payload)
	if err != nil {
		return vpb.Vote{}, err
	}

	if payload.Purpose() == VOTE_PURPOSE_ID {
		var vote vpb.Vote
		err := proto.Unmarshal([]byte(payload.Data()), &vote)
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
