package proxdag

import (
	"context"
	"fmt"
	"sync"

	"github.com/cockroachdb/errors"
	"github.com/iotaledger/hive.go/serix"
	"github.com/iotaledger/hive.go/stringify"

	"github.com/iotaledger/goshimmer/packages/tangle/payload"
)

func init() {
	err := serix.DefaultAPI.RegisterTypeSettings(Payload{}, serix.TypeSettings{}.WithObjectType(uint32(new(Payload).Type())))
	if err != nil {
		panic(fmt.Errorf("error registering ProxDAG type settings: %w", err))
	}
	err = serix.DefaultAPI.RegisterInterfaceObjects((*payload.Payload)(nil), new(Payload))
	if err != nil {
		panic(fmt.Errorf("error registering ProxDAG as Payload interface: %w", err))
	}
}

// NewProxdag creates a new Proxdag.
func NewProxdag() *Proxdag {
	return &Proxdag{
		Events: newEvents(),
	}
}

// Proxdag manages proxdag messages happening over the Tangle.
type Proxdag struct {
	*Events
}

const (
	// PayloadName defines the name of the proxdag payload.
	PayloadName = "proxdag"
	payloadType = 787
)

// Payload represents the proxdag payload type.
type Payload struct {
	Purpose uint32 `serix:"0"`
	Data    string `serix:"1,lengthPrefixType=uint32"`

	bytes      []byte
	bytesMutex sync.RWMutex
}

// NewPayload creates a new proxdag payload.
func NewPayload(purpose uint32, data string) *Payload {
	return &Payload{
		Purpose: purpose,
		Data:    data,
	}
}

// FromBytes parses the marshaled version of a Payload into a Go object.
// It either returns a new Payload or fills an optionally provided Payload with the parsed information.
func FromBytes(bytes []byte) (payloadDecoded *Payload, consumedBytes int, err error) {
	payloadDecoded = new(Payload)

	consumedBytes, err = serix.DefaultAPI.Decode(context.Background(), bytes, payloadDecoded, serix.WithValidation())
	if err != nil {
		err = errors.Errorf("failed to parse Proxdag Payload: %w", err)
		return
	}
	payloadDecoded.bytes = bytes

	return
}

// Bytes returns a marshaled version of this Payload.
func (p *Payload) Bytes() []byte {
	p.bytesMutex.Lock()
	defer p.bytesMutex.Unlock()
	if objBytes := p.bytes; objBytes != nil {
		return objBytes
	}

	objBytes, err := serix.DefaultAPI.Encode(context.Background(), p, serix.WithValidation())
	if err != nil {
		// TODO: what do?
		panic(err)
	}
	p.bytes = objBytes
	return objBytes
}

// String returns a human-friendly representation of the Payload.
func (p *Payload) String() string {
	return stringify.Struct("ProxdagPayload",
		stringify.StructField("purpose", p.Purpose),
		stringify.StructField("data", p.Data),
	)
}

// Type represents the identifier which addresses the proxdag payload type.
var Type = payload.NewType(payloadType, PayloadName)

// Type returns the type of the Payload.
func (p *Payload) Type() payload.Type {
	return Type
}
