package main

import (
	"errors"
	"math"
)

const (
	THRESHOLD  = 0.1
	DECAY_RATE = 0.0001
	DELTA      = 0.01
	K          = 10
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

// GetClients returns a list of clients' names from the
// updates shared in IOTA
func GetClients(updates []mupb.ModelUpdate) []string {
	panic("todo :)")
}

// GetModelUpdates returns a map of clients' names and their updates
// from the set of model updates published in IOTA for the modelID
// NOTE: it returns the updates pending from the latest verification
func GetModelUpdates(modelID string) map[string][]mupb.ModelUpdate {
	panic("todo :)")
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
	updates := GetModelUpdates(modelID)
	clients := GetClients(updates)

	for i := 0; i < len(clients); i++ {
		for j := 0; j < len(clients); j++ {
			if i == j {
				continue
			}
			// todo(lekssays): check multiple updates of the same client
			// get the latest one for the moment
			a := updates[clients[i]]
			b := updates[clients[j]]
			csMatrix[i][j], _ = ComputeCS(a, b)
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
