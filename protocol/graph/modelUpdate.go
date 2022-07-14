package graph

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/Lekssays/ProxDAG/protocol/plugins/proxdag"
	mupb "github.com/Lekssays/ProxDAG/protocol/proto/modelUpdate"
	"github.com/go-redis/redis/v8"
	"github.com/golang/protobuf/proto"
	"github.com/iotaledger/goshimmer/client"
	"google.golang.org/protobuf/encoding/prototext"
)

const (
	MODEL_UPDATE_PYTHON_PURPOSE_ID = 16
	MODEL_UPDATE_GOLANG_PURPOSE_ID = 17
)

type Model struct {
	ID      string
	Updates []string
}

type Response struct {
	MessageID string `json:"messageID,omitempty"`
	Error     string `json:"error,omitempty"`
}

type Message struct {
	Purpose uint32 `json:"purpose"`
	Data    []byte `json:"data"`
}

type Peers struct {
	Peers []Peer `json:"peers"`
}

type Peer struct {
	Pubkey string `json:"pubkey"`
	ID     string `json:"id"`
	Name   string `json:"name"`
}

func GetModelUpdate(messageID string) (mupb.ModelUpdate, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	messageRaw, _ := goshimAPI.GetMessage(messageID)
	payload := new(proxdag.Payload)
	err := payload.FromBytes(messageRaw.Payload)
	if err != nil {
		return mupb.ModelUpdate{}, err
	}

	var mupdate mupb.ModelUpdate
	if payload.Purpose() == MODEL_UPDATE_GOLANG_PURPOSE_ID {
		err = proto.Unmarshal([]byte(payload.Data()), &mupdate)
		if err != nil {
			return mupb.ModelUpdate{}, err
		}
		return mupdate, nil
	} else if payload.Purpose() == MODEL_UPDATE_PYTHON_PURPOSE_ID {
		err = prototext.Unmarshal([]byte(payload.Data()), &mupdate)
		if err != nil {
			return mupb.ModelUpdate{}, err
		}
		return mupdate, nil
	}

	return mupb.ModelUpdate{}, errors.New("PurposeID does not match ModelUpdate Type")

}

func AddModelUpdateEdge(messageID string, graph Graph) (bool, error) {
	mupdate, err := GetModelUpdate(messageID)
	if err != nil {
		return false, err
	}

	graph.AddNode(Node{MessageID: messageID})

	for i := 0; i < len(mupdate.Parents); i++ {
		graph.AddEdge(Node{MessageID: mupdate.Parents[i]}, Node{MessageID: messageID})
	}

	return true, nil
}

func SaveModelUpdate(messageID string, modelUpdate mupb.ModelUpdate) error {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	modelUpdateBytes, err := proto.Marshal(&modelUpdate)
	if err != nil {
		return err
	}

	err = rdb.Set(ctx, messageID, modelUpdateBytes, 0).Err()
	if err != nil {
		return err
	}

	key := fmt.Sprintf("%s!MU!", modelUpdate.ModelID)
	err = rdb.SAdd(ctx, key, messageID).Err()
	if err != nil {
		return err
	}

	err = rdb.Close()
	if err != nil {
		return err
	}

	return nil
}

func RetrieveModelUpdate(modelID string, messageID string) (*mupb.ModelUpdate, error) {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	data, err := rdb.Get(ctx, messageID).Result()
	if err != nil {
		return &mupb.ModelUpdate{}, err
	}

	err = rdb.Close()
	if err != nil {
		return &mupb.ModelUpdate{}, err
	}

	var modelUpdate mupb.ModelUpdate
	err = proto.Unmarshal([]byte(data), &modelUpdate)
	if err != nil {
		return &mupb.ModelUpdate{}, err
	}

	return &modelUpdate, nil
}

func SendModelUpdate(mupdate mupb.ModelUpdate) (string, error) {
	url := GOSHIMMER_NODE + "/proxdag"

	modelUpdateBytes, err := proto.Marshal(&mupdate)
	if err != nil {
		return "", err
	}

	payload := Message{
		Purpose: MODEL_UPDATE_GOLANG_PURPOSE_ID,
		Data:    modelUpdateBytes,
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

	var response Response
	json.Unmarshal(body, &response)
	if strings.Contains(message, "messageID") {
		err = SaveModelUpdate(response.MessageID, mupdate)
		if err != nil {
			return "", err
		}
		return response.MessageID, nil
	}

	return "", errors.New(response.Error)
}

func GetModelUpdatesMessageIDs(modelID string) ([]string, error) {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	key := fmt.Sprintf("%s!MU!", modelID)
	messageIDs, err := rdb.SMembers(ctx, key).Result()
	if err != nil {
		return []string{}, err
	}

	err = rdb.Close()
	if err != nil {
		return []string{}, err
	}

	return messageIDs, nil
}

func GetModelUpdates(modelID string) ([]*mupb.ModelUpdate, error) {
	messageIDs, err := GetModelUpdatesMessageIDs(modelID)
	if err != nil {
		return []*mupb.ModelUpdate{}, err
	}

	var modelUpdates []*mupb.ModelUpdate
	for i := 0; i < len(messageIDs); i++ {
		modelUpdate, err := RetrieveModelUpdate(modelID, messageIDs[i])
		if err != nil {
			return []*mupb.ModelUpdate{}, err
		}
		modelUpdates = append(modelUpdates, modelUpdate)
	}

	return modelUpdates, nil
}

func StoreClientID(id uint32, pubkey string, modelID string) error {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	key := fmt.Sprintf("%s!CL!%s", modelID, pubkey)
	err := rdb.Set(ctx, key, id, 0).Err()
	if err != nil {
		return err
	}

	key = fmt.Sprintf("%s!CL!%d", modelID, id)
	err = rdb.Set(ctx, key, pubkey, 0).Err()
	if err != nil {
		return err
	}

	err = rdb.Close()
	if err != nil {
		return err
	}

	return nil
}

func GetClients(modelID string) ([]string, error) {
	if len(os.Getenv("ENVIRONMENT")) == 0 {
		return []string{}, errors.New("Initialize ENVIRONMENT")
	}

	if os.Getenv("ENVIRONMENT") == "DEV" {
		jsonFile, err := os.Open("./consensus/peers.json")
		if err != nil {
			return []string{}, err
		}

		defer jsonFile.Close()

		byteValue, _ := ioutil.ReadAll(jsonFile)

		var peers Peers
		json.Unmarshal(byteValue, &peers)

		var pubkeys []string
		for i := 0; i < len(peers.Peers); i++ {
			pubkeys = append(pubkeys, peers.Peers[i].Pubkey)
			ID, err := strconv.Atoi(string(peers.Peers[i].ID))
			if err != nil {
				return []string{}, err
			}

			err = StoreClientID(uint32(ID), peers.Peers[i].Pubkey, modelID)
			if err != nil {
				return []string{}, err
			}
		}
		return pubkeys, nil
	} else if os.Getenv("ENVIRONMENT") == "PROD" {
		messageIDs, err := GetModelUpdatesMessageIDs(modelID)

		if err != nil {
			return []string{}, err
		}

		set := make(map[string]bool)
		var clients []string
		for i := 0; i < len(messageIDs); i++ {
			modelUpdate, err := RetrieveModelUpdate(modelID, messageIDs[i])
			if err != nil {
				return []string{}, err
			}
			_, exists := set[modelUpdate.Pubkey]
			if !exists {
				clients = append(clients, modelUpdate.Pubkey)
				set[modelUpdate.Pubkey] = true
			}
		}

		return clients, nil
	}

	return []string{}, errors.New("Invalid operation!")
}

func GetClientID(pubkey string, modelID string) (uint32, error) {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	key := fmt.Sprintf("%s!CL!%s", modelID, pubkey)
	data, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return 0, err
	}

	err = rdb.Close()
	if err != nil {
		return 0, err
	}
	
	ID, err := strconv.Atoi(string(data))
	if err != nil {
		return 0, err
	}

	return uint32(ID), nil
}

func GetClientPubkey(id int, modelID string) (string, error) {
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     "0.0.0.0:6379",
		Password: "",
		DB:       0,
	})

	key := fmt.Sprintf("%s!CL!%d", modelID, id)
	data, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return "", err
	}

	err = rdb.Close()
	if err != nil {
		return "", err
	}

	return string(data), nil
}
