package proxdag

import (
	"time"

	"github.com/iotaledger/hive.go/generics/event"
)

// Events define events occurring within a Proxdag payload.
type Events struct {
	MessageReceived *event.Event[*MessageReceivedEvent]
}

// newEvents returns a new Events object.
func newEvents() (new *Events) {
	return &Events{
		MessageReceived: event.New[*MessageReceivedEvent](),
	}
}

// Event defines the information passed when a proxdag event fires.
type MessageReceivedEvent struct {
	Purpose   uint32
	Data      string
	Timestamp time.Time
	MessageID string
}
