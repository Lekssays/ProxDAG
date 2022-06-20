package consensus

import (
	"bytes"
	"encoding/json"
	"errors"
	"io/ioutil"
	"math"
	"net/http"
	"strings"

	scpb "github.com/Lekssays/ProxDAG/protocol/consensus/proto/score"
	"github.com/Lekssays/ProxDAG/protocol/graph"
	"github.com/Lekssays/ProxDAG/protocol/plugins/proxdag"
	"github.com/golang/protobuf/proto"
	"github.com/iotaledger/goshimmer/client"
	"github.com/sbinet/npyio"
	"github.com/syndtr/goleveldb/leveldb"
)

const (
	THRESHOLD             = 0.1
	DECAY_RATE            = 0.0001
	DELTA                 = 0.01
	K                     = 10
	TRUST_PURPOSE_ID      = 21
	SIMILARITY_PURPOSE_ID = 22
	IPFS_ENDPOINT         = "http://0.0.0.0:5001/api/v0"
)

// ComputeCS returns the cosine similarity of two vectors
func ComputeCS(a []float64, b []float64) (float64, error) {
	count := 0
	length_a := len(a)
	length_b := len(b)
	if length_a > length_b {
		count = length_a
	} else {
		count = length_b
	}
	sumA := 0.0
	s1 := 0.0
	s2 := 0.0
	for k := 0; k < count; k++ {
		if k >= length_a {
			s2 += math.Pow(b[k], 2)
			continue
		}
		if k >= length_b {
			s1 += math.Pow(a[k], 2)
			continue
		}
		sumA += a[k] * b[k]
		s1 += math.Pow(a[k], 2)
		s2 += math.Pow(b[k], 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0, errors.New("Vectors should not be null (all zeros)")
	}

	cosine := sumA / (math.Sqrt(s1) * math.Sqrt(s2))
	return cosine, nil
}

func getAverage(a []float64) float64 {
	sum := 0.0
	for i := 0; i < len(a); i++ {
		sum += a[i]
	}
	return (float64(sum) / float64(len(a)))
}

func ComputeCSMatrix(modelID string) ([][]float64, []float64) {
	var csMatrix [][]float64
	var algnScore []float64
	updates, err := graph.GetModelUpdates(modelID)
	if err != nil {
		return [][]float64{}, []float64{}
	}

	clients, err := graph.GetClients(modelID)
	if err != nil {
		return [][]float64{}, []float64{}
	}

	for i := 0; i < len(clients); i++ {
		for j := 0; j < len(clients); j++ {
			if i == j {
				csMatrix[i][j] = 1.0
				csMatrix[j][i] = 1.0
				continue
			}
			// todo(lekssays): check multiple updates of the same client
			// get the latest one for the moment
			clientAID, err := graph.GetClientID(clients[i], modelID)
			if err != nil {
				return [][]float64{}, []float64{}
			}

			clientBID, err := graph.GetClientID(clients[j], modelID)
			if err != nil {
				return [][]float64{}, []float64{}
			}

			a, err := ToVector(updates[clientAID].Gradients)
			if err != nil {
				return [][]float64{}, []float64{}
			}

			b, err := ToVector(updates[clientBID].Gradients)
			if err != nil {
				return [][]float64{}, []float64{}
			}

			csMatrix[i][j], err = ComputeCS(a, b)
			if err != nil {
				return [][]float64{}, []float64{}
			}
			csMatrix[j][i] = csMatrix[i][j]
		}
		// warning(lekssays): not sure about the alignment score
		algnScore[i] = getAverage(csMatrix[i])
	}
	return csMatrix, algnScore
}

func EvaluatePardoning(clients []string, algnScore []float64, csMatrix [][]float64) ([][]float64, []float64) {
	for i := 0; i < len(clients); i++ {
		for j := 0; j < len(clients); j++ {
			if algnScore[i] > algnScore[j] {
				csMatrix[i][j] *= math.Min(1, algnScore[i]/algnScore[j])
				csMatrix[j][i] = csMatrix[i][j]
			}
		}
		// warning(lekssays): not sure about the alignment score
		algnScore[i] = getAverage(csMatrix[i])
	}
	return csMatrix, algnScore
}

func UpdateTrustScores(clients []string, algnScore []float64) map[string]float32 {
	var trustScores map[string]float32
	for i := 0; i < len(clients); i++ {
		trustScores[clients[i]] *= (1 - DECAY_RATE)
		if algnScore[i] >= THRESHOLD {
			trustScores[clients[i]] -= DELTA
		} else {
			trustScores[clients[i]] += DELTA
		}
	}
	return trustScores
}

func StoreSimilarity(csMatrix scpb.Similarity) error {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return err
	}
	defer db.Close()

	csMatrixBytes, err := proto.Marshal(&csMatrix)
	if err != nil {
		return err
	}

	err = db.Put([]byte("similarity"), csMatrixBytes, nil)
	if err != nil {
		return err
	}

	return nil
}

func StoreTrust(trustScores scpb.Trust) error {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return err
	}
	defer db.Close()

	trustScoresBytes, err := proto.Marshal(&trustScores)
	if err != nil {
		return err
	}

	err = db.Put([]byte("trust"), trustScoresBytes, nil)
	if err != nil {
		return err
	}

	return nil
}

func RetrieveSimilarity() (*scpb.Similarity, error) {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return &scpb.Similarity{}, err
	}
	defer db.Close()

	data, err := db.Get([]byte("similarity"), nil)
	if err != nil {
		return &scpb.Similarity{}, err
	}

	similarity := &scpb.Similarity{}
	err = proto.Unmarshal(data, similarity)
	if err != nil {
		return &scpb.Similarity{}, err
	}

	return similarity, nil
}

func RetrieveTrust() (*scpb.Trust, error) {
	db, err := leveldb.OpenFile(LEVELDB_ENDPOINT, nil)
	if err != nil {
		return &scpb.Trust{}, err
	}
	defer db.Close()

	data, err := db.Get([]byte("trust"), nil)
	if err != nil {
		return &scpb.Trust{}, err
	}

	trust := &scpb.Trust{}
	err = proto.Unmarshal(data, trust)
	if err != nil {
		return &scpb.Trust{}, err
	}

	return trust, nil
}

func PublishSimilarity(similarity scpb.Similarity) (string, error) {
	url := GOSHIMMER_NODE + "/proxdag"

	payload := Message{
		Purpose: SIMILARITY_PURPOSE_ID,
		Data:    []byte(similarity.String()),
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
	var response graph.Response
	json.Unmarshal(body, &response)
	if strings.Contains(message, "messageID") {
		return response.MessageID, nil
	}

	return "", errors.New(response.Error)
}

func PublishTrustScore(trustScores scpb.Trust) (string, error) {
	url := GOSHIMMER_NODE + "/proxdag"

	payload := Message{
		Purpose: TRUST_PURPOSE_ID,
		Data:    []byte(trustScores.String()),
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

func GetSimilarity(messageID string) (scpb.Similarity, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	messageRaw, _ := goshimAPI.GetMessage(messageID)
	payload := new(proxdag.Payload)
	err := payload.FromBytes(messageRaw.Payload)
	if err != nil {
		return scpb.Similarity{}, err
	}

	if payload.Purpose() == SIMILARITY_PURPOSE_ID {
		var similarity scpb.Similarity
		err := proto.Unmarshal([]byte(payload.Data()), &similarity)
		if err != nil {
			return scpb.Similarity{}, err
		}
		return similarity, nil
	}

	return scpb.Similarity{}, errors.New("Unknown payload type!")
}

func GetTrust(messageID string) (scpb.Trust, error) {
	goshimAPI := client.NewGoShimmerAPI(GOSHIMMER_NODE)
	messageRaw, _ := goshimAPI.GetMessage(messageID)
	payload := new(proxdag.Payload)
	err := payload.FromBytes(messageRaw.Payload)
	if err != nil {
		return scpb.Trust{}, err
	}

	if payload.Purpose() == TRUST_PURPOSE_ID {
		var trust scpb.Trust
		err := proto.Unmarshal([]byte(payload.Data()), &trust)
		if err != nil {
			return scpb.Trust{}, err
		}
		return trust, nil
	}

	return scpb.Trust{}, errors.New("Unknown payload type!")
}

func GetContentIPFS(path string) ([]byte, error) {
	url := IPFS_ENDPOINT + "/get?arg=" + path
	req, err := http.NewRequest("POST", url, bytes.NewBuffer([]byte{}))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return []byte{}, err
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)
	return body, nil
}

func AddContentIPFS(content []byte) (string, error) {
	url := IPFS_ENDPOINT + "/add"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(content))

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)
	response := string(body)
	return response, nil
}

func ToVector(GradientsPath string) ([]float64, error) {
	gradientsBytes, err := GetContentIPFS(GradientsPath)
	if err != nil {
		return []float64{}, err
	}

	r, err := npyio.NewReader(bytes.NewReader(gradientsBytes))
	if err != nil {
		return []float64{}, err
	}

	var gradients []float64
	err = r.Read(&gradients)
	if err != nil {
		return []float64{}, err
	}

	return gradients, nil
}
