package modelupdate

import (
	"net/http"

	"github.com/iotaledger/goshimmer/packages/jsonmodels"
	"github.com/labstack/echo"
)

const (
	maxBaseLength    = 32
	parentsMaxLength = 380
	maxContentLength = 4096
)

func configureWebAPI() {
	deps.Server.POST("modelUpdate", SendModelUpdateMessage)
}

// SendModelUpdateMessage sends a modelUpdate message.
func SendModelUpdateMessage(c echo.Context) error {
	req := &Request{}
	if err := c.Bind(req); err != nil {
		return c.JSON(http.StatusBadRequest, jsonmodels.NewErrorResponse(err))
	}

	if len(req.ModelID) > maxBaseLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "ModelID is too long"})
	}
	if len(req.Parents) > parentsMaxLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "Parents is too long"})
	}

	if len(req.Content) > maxContentLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "Content is too long"})
	}
	if len(req.Endpoint) > maxBaseLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "Endpoint is too long"})
	}

	modelUpdatePayload := NewPayload(req.ModelID, req.Parents, req.Content, req.Endpoint)
	msg, err := deps.Tangle.IssuePayload(modelUpdatePayload)
	if err != nil {
		return c.JSON(http.StatusBadRequest, Response{Error: err.Error()})
	}

	return c.JSON(http.StatusOK, Response{MessageID: msg.ID().Base58()})
}

// Request defines the modelUpdate message to send.
type Request struct {
	ModelID  string  `json:"modelID"`
	Parents  []uint8 `json:"parents"`
	Content  []uint8 `json:"content"`
	Endpoint string  `json:"endpoint"`
}

// Response contains the ID of the message sent.
type Response struct {
	MessageID string `json:"messageID,omitempty"`
	Error     string `json:"error,omitempty"`
}
