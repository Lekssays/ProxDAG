package vote

import (
	"github.com/iotaledger/goshimmer/packages/tangle"
	"github.com/iotaledger/hive.go/events"
	"github.com/iotaledger/hive.go/node"
	"github.com/labstack/echo"
	"go.uber.org/dig"
)

const (
	PluginName = "ProxDAGVote"
)

var (
	Plugin *node.Plugin
	deps   = new(dependencies)
)

func init() {
	Plugin = node.NewPlugin(PluginName, deps, node.Enabled, configure)
	Plugin.Events.Init.Attach(events.NewClosure(func(_ *node.Plugin, container *dig.Container) {
		if err := container.Provide(NewVote); err != nil {
			Plugin.Panic(err)
		}
	}))
}

type dependencies struct {
	dig.In
	Tangle *tangle.Tangle
	Server *echo.Echo
	Vote   *Vote
}

func configure(_ *node.Plugin) {
	deps.Tangle.Booker.Events.MessageBooked.Attach(events.NewClosure(onReceiveVoteFromMessageLayer))
	configureWebAPI()
}

func onReceiveVoteFromMessageLayer(messageID tangle.MessageID) {
	var voteEvent *Event
	deps.Tangle.Storage.Message(messageID).Consume(func(message *tangle.Message) {
		if message.Payload().Type() != Type {
			return
		}

		votePayload, _, err := FromBytes(message.Payload().Bytes())
		if err != nil {
			Plugin.LogError(err)
			return
		}

		voteEvent = &Event{
			ModelID:   votePayload.ModelID,
			VoteID:    votePayload.VoteID,
			Decision:  votePayload.Decision,
			Metadata:  votePayload.Metadata,
			Timestamp: message.IssuingTime(),
			MessageID: message.ID().Base58(),
		}
	})

	if voteEvent == nil {
		return
	}

	deps.Vote.Events.MessageReceived.Trigger(voteEvent)
}
