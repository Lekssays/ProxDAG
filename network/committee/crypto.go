package main

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"os"

	redis "github.com/go-redis/redis/v8"
)

func GenerateRSAKeyPair() (*rsa.PrivateKey, *rsa.PublicKey) {
	privkey, _ := rsa.GenerateKey(rand.Reader, 4096)
	return privkey, &privkey.PublicKey
}

func SaveKey(content string, key string) (bool, error) {
	var ctx = context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     REDIS_ENDPOINT,
		Password: "",
		DB:       0,
	})

	err := rdb.Set(ctx, key, content, 0).Err()
	if err != nil {
		return false, err
	}

	return true, nil
}

func GetKey(key string) (string, error) {
	var ctx = context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     REDIS_ENDPOINT,
		Password: "",
		DB:       0,
	})

	content, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return "", err
	}

	return content, nil
}

func SaveKeyBytes(content []byte, key string) (bool, error) {
	var ctx = context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     REDIS_ENDPOINT,
		Password: "",
		DB:       0,
	})

	err := rdb.Set(ctx, key, content, 0).Err()
	if err != nil {
		return false, err
	}

	return true, nil
}

func GetKeyBytes(key string) ([]byte, error) {
	var ctx = context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr:     REDIS_ENDPOINT,
		Password: "",
		DB:       0,
	})

	content, err := rdb.Get(ctx, key).Result()
	if err != nil {
		return []byte{}, err
	}

	return []byte(content), nil
}

func WriteToFile(content string, filename string) (bool, error) {
	f, err := os.Create(filename)
	if err != nil {
		return false, err
	}
	defer f.Close()
	_, e := f.WriteString(content)
	return true, e
}

func ExportRSAPrivateKey(privkey *rsa.PrivateKey) string {
	privkey_bytes := x509.MarshalPKCS1PrivateKey(privkey)
	privkey_pem := pem.EncodeToMemory(
		&pem.Block{
			Type:  "RSA PRIVATE KEY",
			Bytes: privkey_bytes,
		},
	)
	ok, _ := WriteToFile(string(privkey_pem), "privkey.pem")
	if !ok {
		return ""
	}
	saved, _ := SaveKey(string(privkey_pem), "privkey")
	if !saved {
		return ""
	}
	return string(privkey_pem)
}

func ParseRSAPrivateKey(privPEM string) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode([]byte(privPEM))
	if block == nil {
		return nil, errors.New("failed to parse PEM block containing the key")
	}

	priv, err := x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	return priv, nil
}

func ExportRSAPublicKey(pubkey *rsa.PublicKey) string {
	pubkey_bytes, err := x509.MarshalPKIXPublicKey(pubkey)
	if err != nil {
		return ""
	}
	pubkey_pem := pem.EncodeToMemory(
		&pem.Block{
			Type:  "RSA PUBLIC KEY",
			Bytes: pubkey_bytes,
		},
	)
	ok, _ := WriteToFile(string(pubkey_pem), "pubkey.pem")
	if !ok {
		return ""
	}

	saved, _ := SaveKey(string(pubkey_pem), "pubkey")
	if !saved {
		return ""
	}

	return string(pubkey_pem)
}

func ParseRSAPublicKey(pubPEM string) (*rsa.PublicKey, error) {
	block, _ := pem.Decode([]byte(pubPEM))
	if block == nil {
		return nil, errors.New("failed to parse PEM block containing the key")
	}

	pub, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	switch pub := pub.(type) {
	case *rsa.PublicKey:
		return pub, nil
	default:
		break
	}
	return nil, errors.New("Key type is not RSA")
}

func Encrypt(message string, publicKey *rsa.PublicKey) []byte {
	encryptedBytes, err := rsa.EncryptOAEP(
		sha256.New(),
		rand.Reader,
		publicKey,
		[]byte(message),
		nil)

	if err != nil {
		panic(err)
	}
	return encryptedBytes
}

func Decrypt(encryptedBytes []byte, privateKey *rsa.PrivateKey) []byte {
	decryptedBytes, err := privateKey.Decrypt(nil, encryptedBytes, &rsa.OAEPOptions{Hash: crypto.SHA256})
	if err != nil {
		panic(err)
	}
	return decryptedBytes
}

func Sign(message string, privateKey rsa.PrivateKey) (string, string) {
	msgHash := sha256.New()
	_, err := msgHash.Write([]byte(message))
	if err != nil {
		panic(err)
	}
	checksum := msgHash.Sum(nil)

	signature, err := rsa.SignPSS(rand.Reader, &privateKey, crypto.SHA256, checksum, nil)
	if err != nil {
		panic(err)
	}

	return base64.StdEncoding.EncodeToString(signature), base64.StdEncoding.EncodeToString(checksum)
}

func Verify(checksum string, signature string, publicKey *rsa.PublicKey) bool {
	decodedSignature, _ := base64.StdEncoding.DecodeString(signature)
	decodedChecksum, _ := base64.StdEncoding.DecodeString(checksum)
	err := rsa.VerifyPSS(publicKey, crypto.SHA256, decodedChecksum, decodedSignature, nil)
	if err != nil {
		fmt.Println("could not verify signature: ", err)
		return false
	}
	return true
}

func HashSHA256(content string) [32]byte {
	return sha256.Sum256([]byte(content))
}
