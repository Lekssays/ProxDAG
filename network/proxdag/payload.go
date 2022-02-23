package main

import (
	"fmt"
	"sync"
	"time"

	"github.com/cockroachdb/errors"
	"github.com/iotaledger/hive.go/events"
	"github.com/iotaledger/hive.go/marshalutil"
	"github.com/iotaledger/hive.go/stringify"

	"github.com/iotaledger/goshimmer/packages/tangle/payload"
)

// NewModelUpdate creates a new ModelUpdate.
func NewModelUpdate() *ModelUpdate {
	return &ModelUpdate{
		Events: Events{
			MessageReceived: events.NewEvent(modelUpdateEventCaller),
		},
	}
}

type ModelUpdate struct {
	Events
}

type Events struct {
	MessageReceived *events.Event
}

type Event struct {
	ModelID string
	ParentA string
	ParentB string
	Content string

	Timestamp time.Time
	MessageID string
}

func modelUpdateEventCaller(handler interface{}, params ...interface{}) {
	handler.(func(*Event))(params[0].(*Event))
}

const (
	PayloadName = "modelUpdate"
	payloadType = 989
)

// Payload represents the modelUpdate payload type.
type Payload struct {
	ModelID    string
	ModelIDLen uint32
	ParentA    string
	ParentALen uint32
	ParentB    string
	ParentBLen uint32
	Content    string
	ContentLen uint32

	bytes      []byte
	bytesMutex sync.RWMutex
}

func NewPayload(modelID string, parentA string, parentB string, content string) *Payload {
	return &Payload{
		ModelID:    modelID,
		ModelIDLen: uint32(len([]byte(modelID))),
		ParentA:    parentA,
		ParentALen: uint32(len([]byte(parentA))),
		ParentB:    parentB,
		ParentBLen: uint32(len([]byte(parentB))),
		Content:    content,
		ContentLen: uint32(len(content)),
	}
}

// FromBytes parses the marshaled version of a Payload into a Go object.
// It either returns a new Payload or fills an optionally provided Payload with the parsed information.
func FromBytes(bytes []byte) (result *Payload, consumedBytes int, err error) {
	marshalUtil := marshalutil.New(bytes)
	result, err = Parse(marshalUtil)
	consumedBytes = marshalUtil.ReadOffset()

	return
}

// Parse unmarshals a Payload using the given marshalUtil (for easier marshaling/unmarshaling).
func Parse(marshalUtil *marshalutil.MarshalUtil) (result *Payload, err error) {
	// read information that are required to identify the payload from the outside
	if _, err = marshalUtil.ReadUint32(); err != nil {
		err = fmt.Errorf("failed to parse payload size of modelUpdate payload: %w", err)
		return
	}
	if _, err = marshalUtil.ReadUint32(); err != nil {
		err = fmt.Errorf("failed to parse payload type of modelUpdate payload: %w", err)
		return
	}

	// parse ModelID
	result = &Payload{}
	modelIDLen, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse modelIDLen field of modelUpdate payload: %w", err)
		return
	}
	result.ModelIDLen = modelIDLen

	modelID, err := marshalUtil.ReadBytes(int(modelIDLen))
	if err != nil {
		err = fmt.Errorf("failed to parse from field of modelUpdate payload: %w", err)
		return
	}
	result.ModelID = string(modelID)

	// parse ParentA
	parentALen, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse parentALen field of modelUpdate payload: %w", err)
		return
	}
	result.ParentALen = parentALen

	parentA, err := marshalUtil.ReadBytes(int(parentALen))
	if err != nil {
		err = fmt.Errorf("failed to parse to field of modelUpdate payload: %w", err)
		return
	}
	result.ParentA = string(parentA)

	// parse ParentB
	parentBLen, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse parentBLen field of modelUpdate payload: %w", err)
		return
	}
	result.ParentBLen = parentBLen

	parentB, err := marshalUtil.ReadBytes(int(parentBLen))
	if err != nil {
		err = fmt.Errorf("failed to parse to field of modelUpdate payload: %w", err)
		return
	}
	result.ParentB = string(parentB)

	// parse Content
	contentLen, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse contentLen field of modelUpdate payload: %w", err)
		return
	}
	result.ContentLen = contentLen

	content, err := marshalUtil.ReadBytes(int(contentLen))
	if err != nil {
		err = fmt.Errorf("failed to parse content field of modelUpdate payload: %w", err)
		return
	}
	result.Content = string(content)

	// store bytes, so we don't have to marshal manually
	consumedBytes := marshalUtil.ReadOffset()
	copy(result.bytes, marshalUtil.Bytes()[:consumedBytes])

	return result, nil
}

// Bytes returns a marshaled version of this Payload.
func (p *Payload) Bytes() (bytes []byte) {
	// acquire lock for reading bytes
	p.bytesMutex.RLock()

	// return if bytes have been determined already
	if bytes = p.bytes; bytes != nil {
		p.bytesMutex.RUnlock()
		return
	}

	// switch to write lock
	p.bytesMutex.RUnlock()
	p.bytesMutex.Lock()
	defer p.bytesMutex.Unlock()

	// return if bytes have been determined in the mean time
	if bytes = p.bytes; bytes != nil {
		return
	}

	payloadLength := int(p.ModelIDLen + p.ParentALen + p.ParentBLen + p.ContentLen + marshalutil.Uint32Size*3)
	// initialize helper
	marshalUtil := marshalutil.New(marshalutil.Uint32Size + marshalutil.Uint32Size + payloadLength)

	// marshal the payload specific information
	marshalUtil.WriteUint32(payload.TypeLength + uint32(payloadLength))
	marshalUtil.WriteBytes(Type.Bytes())
	marshalUtil.WriteUint32(p.ModelIDLen)
	marshalUtil.WriteBytes([]byte(p.ModelID))
	marshalUtil.WriteUint32(p.ParentALen)
	marshalUtil.WriteBytes([]byte(p.ParentA))
	marshalUtil.WriteUint32(p.ParentBLen)
	marshalUtil.WriteBytes([]byte(p.ParentB))
	marshalUtil.WriteUint32(p.ContentLen)
	marshalUtil.WriteBytes([]byte(p.Content))

	bytes = marshalUtil.Bytes()

	return bytes
}

// String returns a human-friendly representation of the Payload.
func (p *Payload) String() string {
	return stringify.Struct("ModelUpdatePayload",
		stringify.StructField("modelID", p.ModelID),
		stringify.StructField("parentA", p.ParentA),
		stringify.StructField("parentB", p.ParentB),
		stringify.StructField("content", p.Content),
	)
}

// Type represents the identifier which addresses the modelUpdate payload type.
var Type = payload.NewType(payloadType, PayloadName, func(data []byte) (payload payload.Payload, err error) {
	var consumedBytes int
	payload, consumedBytes, err = FromBytes(data)
	if err != nil {
		return nil, err
	}
	if consumedBytes != len(data) {
		return nil, errors.New("not all payload bytes were consumed")
	}
	return
})

// Type returns the type of the Payload.
func (p *Payload) Type() payload.Type {
	return Type
}
