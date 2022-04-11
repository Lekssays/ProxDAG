package proxdag

import (
	"github.com/iotaledger/goshimmer/packages/tangle"
	"github.com/iotaledger/hive.go/events"
	"github.com/iotaledger/hive.go/node"
	"github.com/labstack/echo"
	"go.uber.org/dig"
)

const (
	PluginName = "ProxDAG"
)

var (
	Plugin *node.Plugin
	deps   = new(dependencies)
)

func init() {
	Plugin = node.NewPlugin(PluginName, deps, node.Enabled, configure)
	Plugin.Events.Init.Attach(events.NewClosure(func(_ *node.Plugin, container *dig.Container) {
		if err := container.Provide(NewProxdag); err != nil {
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
	deps.Tangle.Booker.Events.MessageBooked.Attach(events.NewClosure(onReceiveProxdagFromMessageLayer))
	configureWebAPI()
}

func onReceiveProxdagFromMessageLayer(messageID tangle.MessageID) {
	var proxdagEvent *Event
	deps.Tangle.Storage.Message(messageID).Consume(func(message *tangle.Message) {
		if message.Payload().Type() != Type {
			return
		}

		proxdagPayload, _, err := FromBytes(message.Payload().Bytes())
		if err != nil {
			Plugin.LogError(err)
			return
		}

		proxdagEvent = &Event{
			Purpose:   proxdagPayload.Purpose,
			Data:      proxdagPayload.Data,
			Timestamp: message.IssuingTime(),
			MessageID: message.ID().Base58(),
		}
	})

	if proxdagEvent == nil {
		return
	}

	deps.Proxdag.Events.MessageReceived.Trigger(proxdagEvent)
}
