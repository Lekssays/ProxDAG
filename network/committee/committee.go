package main

import (
	"context"
	"fmt"
	peerpb "github.com/Lekssays/ProxDAG/network/committee/proto/peer"
	"github.com/ProtonMail/go-ecvrf/ecvrf"
	"github.com/go-redis/redis/v8"
	"github.com/golang/protobuf/proto"
	"log"
)

const (
	GOSHIMMER_NODE = "http://0.0.0.0:8080"
	REDIS_ENDPOINT = "http://127.0.0.1:6379"
)

func SavePeer(peer peerpb.Peer) (bool, error) {
	var ctx = context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     REDIS_ENDPOINT,
		Password: "",
		DB:       0,
	})

	peerString := proto.MarshalTextString(&peer)

	err := rdb.SAdd(ctx, "peers", peerString).Err()
	if err != nil {
		return false, err
	}

	return true, nil
}

func GetPeers() ([]peerpb.Peer, error) {
	var ctx = context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     REDIS_ENDPOINT,
		Password: "",
		DB:       0,
	})

	peersStrings, err := rdb.SMembers(ctx, "peers").Result()
	if err != nil {
		return []peerpb.Peer{}, err
	}

	var peers []peerpb.Peer
	for i := 0; i < len(peersStrings); i++ {
		var peer peerpb.Peer
		err = proto.UnmarshalText(peersStrings[i], &peer)
		if err != nil {
			log.Println(err.Error())
		}
		peers = append(peers, peer)
	}

	return peers, nil
}

func main() {
	fmt.Println("Hello :) Dynamic Committee")

	// Keygen
	secretKey, err := ecvrf.GenerateKey(nil)
	verificationKey, err := secretKey.Public()

	SecretKeyBin := secretKey.Bytes()
	VerificationKeyBin := verificationKey.Bytes()

	// Prove
	secretKey, err = ecvrf.NewPrivateKey(SecretKeyBin)
	message := []byte("alice")
	y, proof, err := secretKey.Prove(message)
	if err != nil {
		fmt.Println(err.Error())
	}

	// Verify
	verificationKey, err = ecvrf.NewPublicKey(VerificationKeyBin)
	verified, verificationVRF, err := verificationKey.Verify(message, proof)
	if err != nil {
		fmt.Println(err.Error())
	}

	if verified {
		fmt.Println("Output is verified :)")
		fmt.Println("y =", y)
		fmt.Println("verificationY =", verificationVRF)
	} else {
		fmt.Println("Output is NOT verified :)")
	}

}
