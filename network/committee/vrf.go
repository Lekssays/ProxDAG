package main

import (
	"context"
	"encoding/hex"
	"github.com/ProtonMail/go-ecvrf/ecvrf"
	"github.com/drand/drand/client"
	"github.com/drand/drand/client/http"
	"log"
)

func GetMagicSeed() ([]byte, error) {
	var urls = []string{
		"https://api.drand.sh",
		"https://drand.cloudflare.com",
	}

	// the hash is for a public entropy league (a set of drand nodes)
	var chainHash, _ = hex.DecodeString("8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce")
	c, err := client.New(
		client.From(http.ForURLs(urls, chainHash)...),
		client.WithChainHash(chainHash),
	)
	if err != nil {
		log.Fatal(err)
		return []byte{}, err
	}
	r, err := c.Get(context.Background(), 0)
	if err != nil {
		log.Fatal(err)
		return []byte{}, err
	}
	return r.Randomness(), nil
}

func GenerateVRFKeys() ([]byte, []byte, error) {
	secretKey, err := ecvrf.GenerateKey(nil)
	if err != nil {
		return []byte{}, []byte{}, err
	}
	SecretKeyBin := secretKey.Bytes()
	SaveKeyBytes(SecretKeyBin, "secretKey")

	verificationKey, err := secretKey.Public()
	if err != nil {
		return []byte{}, []byte{}, err
	}
	VerificationKeyBin := verificationKey.Bytes()
	SaveKeyBytes(VerificationKeyBin, "verificationKey")

	return SecretKeyBin, VerificationKeyBin, nil
}

func Prove(message []byte) ([]byte, []byte, error) {
	SecretKeyBin, err := GetKeyBytes("secretKey")
	if err != nil {
		return []byte{}, []byte{}, err
	}

	secretKey, err := ecvrf.NewPrivateKey(SecretKeyBin)
	if err != nil {
		return []byte{}, []byte{}, err
	}

	y, proof, err := secretKey.Prove(message)
	if err != nil {
		return []byte{}, []byte{}, err
	}

	return y, proof, nil
}

func VerifyVRF(message []byte, proof []byte) (bool, error) {
	VerificationKeyBin, err := GetKeyBytes("verificationKey")
	if err != nil {
		return false, err
	}

	verificationKey, err := ecvrf.NewPublicKey(VerificationKeyBin)
	if err != nil {
		return false, err
	}

	verified, _, err := verificationKey.Verify(message, proof)
	if err != nil {
		return false, err
	}

	return verified, nil
}
