package proxdag

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

// NewProxdag creates a new Proxdag.
func NewProxdag() *Proxdag {
	return &Proxdag{
		Events: Events{
			MessageReceived: events.NewEvent(proxdagEventCaller),
		},
	}
}

type Proxdag struct {
	Events
}

type Events struct {
	MessageReceived *events.Event
}

type Event struct {
	Purpose uint32
	Data    string

	Timestamp time.Time
	MessageID string
}

func proxdagEventCaller(handler interface{}, params ...interface{}) {
	handler.(func(*Event))(params[0].(*Event))
}

const (
	PayloadName = "proxdag"
	payloadType = 787
)

// Payload represents the proxdag payload type.
type Payload struct {
	Purpose    uint32
	Data       string
	DataLen    uint32

	bytes      []byte
	bytesMutex sync.RWMutex
}

func NewPayload(purpose uint32, data string) *Payload {
	return &Payload{
		Purpose:    purpose,
		Data:       data,
		DataLen:    uint32(len([]byte(data))),
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
		err = fmt.Errorf("failed to parse payload size of proxdag payload: %w", err)
		return
	}
	if _, err = marshalUtil.ReadUint32(); err != nil {
		err = fmt.Errorf("failed to parse payload size of proxdag payload: %w", err)
		return
	}

	// parse puprose
	result = &Payload{}
	puprose, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse Purpose field of proxdag payload: %w", err)
		return
	}
	result.Purpose = puprose

	// parse Data
	result = &Payload{}
	dataLen, err := marshalUtil.ReadUint32()
	if err != nil {
		err = fmt.Errorf("failed to parse DataLen field of proxdag payload: %w", err)
		return
	}
	result.DataLen = dataLen

	data, err := marshalUtil.ReadBytes(int(dataLen))
	if err != nil {
		err = fmt.Errorf("failed to parse Data field of proxdag payload: %w", err)
		return
	}
	result.Data = string(data)

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

	payloadLength := int(p.DataLen + marshalutil.Uint32Size*2)

	// initialize helper
	marshalUtil := marshalutil.New(marshalutil.Uint32Size + marshalutil.Uint32Size + payloadLength)

	// marshal the payload specific information
	marshalUtil.WriteUint32(payload.TypeLength + uint32(payloadLength))
	marshalUtil.WriteBytes(Type.Bytes())
	marshalUtil.WriteUint32(p.Purpose)
	marshalUtil.WriteUint32(p.DataLen)
	marshalUtil.WriteBytes([]byte(p.Data))

	bytes = marshalUtil.Bytes()

	return bytes
}

// String returns a human-friendly representation of the Payload.
func (p *Payload) String() string {
	return stringify.Struct("ProxdagPayload",
		stringify.StructField("purpose", p.Purpose),
		stringify.StructField("data", p.Data),
	)
}

// Type represents the identifier which addresses the proxdag payload type.
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
