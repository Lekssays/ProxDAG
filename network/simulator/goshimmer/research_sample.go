package plugins

import (
	"github.com/iotaledger/hive.go/node"

	"proxdag" // add your plugin to GOROOT (or to a public github repo to use go get)

	"github.com/iotaledger/goshimmer/plugins/activity"
	analysisclient "github.com/iotaledger/goshimmer/plugins/analysis/client"
	analysisdashboard "github.com/iotaledger/goshimmer/plugins/analysis/dashboard"
	analysisserver "github.com/iotaledger/goshimmer/plugins/analysis/server"
	"github.com/iotaledger/goshimmer/plugins/chat"
	"github.com/iotaledger/goshimmer/plugins/networkdelay"
	"github.com/iotaledger/goshimmer/plugins/prometheus"
	"github.com/iotaledger/goshimmer/plugins/remotelog"
	"github.com/iotaledger/goshimmer/plugins/remotemetrics"
	"github.com/iotaledger/goshimmer/plugins/txstream"
)

// Research contains research plugins of a GoShimmer node.
var Research = node.Plugins(
	remotelog.Plugin,
	analysisserver.Plugin,
	analysisclient.Plugin,
	analysisdashboard.Plugin,
	prometheus.Plugin,
	remotemetrics.Plugin,
	networkdelay.App(),
	txstream.Plugin,
	activity.Plugin,
	chat.Plugin,
	proxdag.Plugin,
)
