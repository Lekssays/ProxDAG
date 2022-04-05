package vote

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

// NewVote creates a new Vote.
func NewVote() *Vote {
	return &Vote{
		Events: Events{
			MessageReceived: events.NewEvent(voteEventCaller),
		},
	}
}

type Vote struct {
	Events
}

type Events struct {
	MessageReceived *events.Event
}

type Event struct {
	ModelID  string
	VoteID   string
	Decision string
	Metadata string

	Timestamp time.Time
	MessageID string
}

func voteEventCaller(handler interface{}, params ...interface{}) {
	handler.(func(*Event))(params[0].(*Event))
}

const (
	PayloadName = "vote"
	payloadType = 786
)

// Payload represents the vote payload type.
type Payload struct {
	ModelID     string
	ModelIDLen  uint32
	VoteID      string
	VoteIDLen   uint32
	Decision    string
	DecisionLen uint32
	Metadata    string
	MetadataLen uint32

	bytes      []byte
	bytesMutex sync.RWMutex
}

func NewPayload(modelID string, voteID string, decision string, metadata string) *Payload {
	return &Payload{
		ModelID:     modelID,
		ModelIDLen:  uint32(len([]byte(modelID))),
		VoteID:      voteID,
		VoteIDLen:   uint32(len([]byte(voteID))),
		Decision:    decision,
		DecisionLen: uint32(len([]byte(decision))),
		Metadata:    metadata,
		MetadataLen: uint32(len([]byte(metadata))),
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
		err = fmt.Errorf("failed to parse payload size of vote payload: %w", err)
		return
	}
	if _, err = marshalUtil.ReadUint32(); err != nil {
		err = fmt.Errorf("failed to parse payload size of vote payload: %w", err)
		return
	}

	// parse ModelID
	result = &Payload{}
	modelIDLen, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse ModelIDLen field of vote payload: %w", err)
		return
	}
	result.ModelIDLen = modelIDLen

	modelID, err := marshalUtil.ReadBytes(int(modelIDLen))
	if err != nil {
		err = fmt.Errorf("failed to parse ModelID field of vote payload: %w", err)
		return
	}
	result.ModelID = string(modelID)

	// parse VoteID
	result = &Payload{}
	voteIDLen, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse VoteIDLen field of vote payload: %w", err)
		return
	}
	result.VoteIDLen = voteIDLen

	voteID, err := marshalUtil.ReadBytes(int(voteIDLen))
	if err != nil {
		err = fmt.Errorf("failed to parse VoteID field of vote payload: %w", err)
		return
	}
	result.VoteID = string(voteID)

	// parse Decision
	result = &Payload{}
	decisionLen, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse DecisionLen field of vote payload: %w", err)
		return
	}
	result.DecisionLen = decisionLen

	decision, err := marshalUtil.ReadBytes(int(decisionLen))
	if err != nil {
		err = fmt.Errorf("failed to parse Decision field of vote payload: %w", err)
		return
	}
	result.Decision = string(decision)

	// parse Metadata
	result = &Payload{}
	metadataLen, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse MetadataLen field of vote payload: %w", err)
		return
	}
	result.MetadataLen = metadataLen

	metadata, err := marshalUtil.ReadBytes(int(metadataLen))
	if err != nil {
		err = fmt.Errorf("failed to parse Metadata field of vote payload: %w", err)
		return
	}
	result.Metadata = string(metadata)

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

	payloadLength := int(p.ModelIDLen + p.VoteIDLen + p.DecisionLen + p.MetadataLen + marshalutil.Uint32Size*4)

	// initialize helper
	marshalUtil := marshalutil.New(marshalutil.Uint32Size + marshalutil.Uint32Size + payloadLength)

	// marshal the payload specific information
	marshalUtil.WriteUint32(payload.TypeLength + uint32(payloadLength))
	marshalUtil.WriteBytes(Type.Bytes())
	marshalUtil.WriteUint32(p.ModelIDLen)
	marshalUtil.WriteBytes([]byte(p.ModelID))
	marshalUtil.WriteUint32(p.VoteIDLen)
	marshalUtil.WriteBytes([]byte(p.VoteID))
	marshalUtil.WriteUint32(p.DecisionLen)
	marshalUtil.WriteBytes([]byte(p.Decision))
	marshalUtil.WriteUint32(p.MetadataLen)
	marshalUtil.WriteBytes([]byte(p.Metadata))

	bytes = marshalUtil.Bytes()

	return bytes
}

// String returns a human-friendly representation of the Payload.
func (p *Payload) String() string {
	return stringify.Struct("VotePayload",
		stringify.StructField("modelID", p.ModelID),
		stringify.StructField("voteID", p.VoteID),
		stringify.StructField("decision", p.Decision),
		stringify.StructField("endpoint", p.Metadata),
	)
}

// Type represents the identifier which addresses the vote payload type.
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
