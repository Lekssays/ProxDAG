package main

import (
	"fmt"
)

const (
	GOSHIMMER_NODE = "http://0.0.0.0:8080"
	REDIS_ENDPOINT = "http://127.0.0.1:6379"
)

func main() {
	fmt.Println("Hello :) Dynamic Committee")

	_, _, err := GenerateVRFKeys()
	if err != nil {
		fmt.Errorf(err.Error())
	}

	message := []byte("ProxDAG is amazing")
	_, proof, err := Prove(message)
	if err != nil {
		fmt.Errorf(err.Error())
	}

	verified, err := VerifyVRF(message, proof)
	if err != nil {
		fmt.Errorf(err.Error())
	}

	if verified {
		fmt.Println("Output is verified :)")
	} else {
		fmt.Println("Output is NOT verified :)")
	}

}
