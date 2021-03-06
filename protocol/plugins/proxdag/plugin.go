package proxdag

import (
	"github.com/labstack/echo"
	"go.uber.org/dig"

	"github.com/iotaledger/hive.go/generics/event"
	"github.com/iotaledger/hive.go/node"

	"github.com/iotaledger/goshimmer/packages/tangle"
)

const (
	// PluginName contains the human-readable name of the plugin.
	PluginName = "ProxDAG"
)

var (
	// Plugin is the "plugin" instance of the proxdag application.
	Plugin *node.Plugin
	deps   = new(dependencies)
)

func init() {
	Plugin = node.NewPlugin(PluginName, deps, node.Enabled, configure)
	Plugin.Events.Init.Hook(event.NewClosure[*node.InitEvent](func(event *node.InitEvent) {
		if err := event.Container.Provide(NewProxdag); err != nil {
			Plugin.Panic(err)
		}
	}))
}

type dependencies struct {
	dig.In
	Tangle  *tangle.Tangle
	Server  *echo.Echo
	Proxdag *Proxdag
}

func configure(_ *node.Plugin) {
	deps.Tangle.Booker.Events.MessageBooked.Attach(event.NewClosure(func(event *tangle.MessageBookedEvent) {
		onReceiveMessageFromMessageLayer(event.MessageID)
	}))
	configureWebAPI()
}

func onReceiveMessageFromMessageLayer(messageID tangle.MessageID) {
	var proxdagEvent *MessageReceivedEvent
	deps.Tangle.Storage.Message(messageID).Consume(func(message *tangle.Message) {
		if message.Payload().Type() != Type {
			return
		}

		proxdagPayload := message.Payload().(*Payload)
		proxdagEvent = &MessageReceivedEvent{
			Purpose:   proxdagPayload.Purpose(),
			Data:      proxdagPayload.Data(),
			Timestamp: message.IssuingTime(),
			MessageID: message.ID().Base58(),
		}
	})

	if proxdagEvent == nil {
		return
	}

	deps.Proxdag.Events.MessageReceived.Trigger(proxdagEvent)
}
