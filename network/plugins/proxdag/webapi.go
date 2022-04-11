package proxdag

import (
	"net/http"

	"github.com/iotaledger/goshimmer/packages/jsonmodels"
	"github.com/labstack/echo"
)

const (
	maxPurposeLen = 10
	maxDataLength = 4096
)

func configureWebAPI() {
	deps.Server.POST("proxdag", SendProxdagMessage)
}

// SendProxdagMessage sends a proxdag message.
func SendProxdagMessage(c echo.Context) error {
	req := &Request{}
	if err := c.Bind(req); err != nil {
		return c.JSON(http.StatusBadRequest, jsonmodels.NewErrorResponse(err))
	}

	if len(req.Purpose) > maxPurposeLen {
		return c.JSON(http.StatusBadRequest, Response{Error: "Purpose is too long"})
	}
	if len(req.Data) > maxDataLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "Data is too long"})
	}

	proxdagPayload := NewPayload(req.Purpose, req.Data)
	msg, err := deps.Tangle.IssuePayload(proxdagPayload)
	if err != nil {
		return c.JSON(http.StatusBadRequest, Response{Error: err.Error()})
	}

	return c.JSON(http.StatusOK, Response{MessageID: msg.ID().Base58()})
}

// Request defines the proxdag message to send.
type Request struct {
	Purpose string `json:"purpose"`
	Data    string `json:"data"`
}

// Response contains the ID of the message sent.
type Response struct {
	MessageID string `json:"messageID,omitempty"`
	Error     string `json:"error,omitempty"`
}
