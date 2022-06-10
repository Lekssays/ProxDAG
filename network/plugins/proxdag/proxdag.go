package proxdag

import (
	"fmt"

	"github.com/iotaledger/hive.go/generics/model"
	"github.com/iotaledger/hive.go/serix"

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

// Payload represents the chat payload type.
type Payload struct {
	model.Immutable[Payload, *Payload, payloadModel] `serix:"0"`
}

type payloadModel struct {
	Purpose uint32 `serix:"0"`
	Data    string `serix:"1,lengthPrefixType=uint32"`
}

// NewPayload creates a new chat payload.
func NewPayload(purpose uint32, data string) *Payload {
	return model.NewImmutable[Payload](&payloadModel{
		Purpose: purpose,
		Data:    data,
	},
	)
}

// Type represents the identifier which addresses the chat payload type.
var Type = payload.NewType(payloadType, PayloadName)

// Type returns the type of the Payload.
func (p *Payload) Type() payload.Type {
	return Type
}

// Purpose returns an author of the message.
func (p *Payload) Purpose() uint32 {
	return p.M.Purpose
}

// Data returns a recipient of the message.
func (p *Payload) Data() string {
	return p.M.Data
}
