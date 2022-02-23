package main

import (
	"github.com/iotaledger/goshimmer/packages/chat"
	"github.com/iotaledger/goshimmer/packages/tangle"
	"github.com/iotaledger/hive.go/events"
	"github.com/iotaledger/hive.go/node"
	"github.com/labstack/echo"
	"go.uber.org/dig"
)

const (
	PluginName = "ProxDag"
)

var (
	Plugin *node.Plugin
	deps   = new(dependencies)
)

func init() {
	Plugin = node.NewPlugin(PluginName, deps, node.Enabled, configure)
	Plugin.Events.Init.Attach(events.NewClosure(func(_ *node.Plugin, container *dig.Container) {
		if err := container.Provide(NewModelUpdate); err != nil {
			Plugin.Panic(err)
		}
	}))
}

type dependencies struct {
	dig.In
	Tangle      *tangle.Tangle
	Server      *echo.Echo
	ModelUpdate *ModelUpdate
}

func configure(_ *node.Plugin) {
	deps.Tangle.Booker.Events.MessageBooked.Attach(events.NewClosure(onReceiveModelUpdateFromMessageLayer))
	configureWebAPI()
}

func onReceiveModelUpdateFromMessageLayer(messageID tangle.MessageID) {
	var modelUpdateEvent *Event
	deps.Tangle.Storage.Message(messageID).Consume(func(message *tangle.Message) {
		if message.Payload().Type() != chat.Type {
			return
		}

		modelUpdatePayload, _, err := FromBytes(message.Payload().Bytes())
		if err != nil {
			Plugin.LogError(err)
			return
		}

		modelUpdateEvent = &Event{
			ModelID:   modelUpdatePayload.ModelID,
			ParentA:   modelUpdatePayload.ParentA,
			ParentB:   modelUpdatePayload.ParentB,
			Content:   modelUpdatePayload.Content,
			Timestamp: message.IssuingTime(),
			MessageID: message.ID().Base58(),
		}
	})

	if modelUpdateEvent == nil {
		return
	}

	deps.ModelUpdate.Events.MessageReceived.Trigger(modelUpdateEvent)
}
