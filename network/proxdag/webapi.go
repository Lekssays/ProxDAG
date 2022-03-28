package proxdag

import (
	"net/http"

	"github.com/iotaledger/goshimmer/packages/jsonmodels"
	"github.com/labstack/echo"
)

const (
	maxBaseLength    = 32
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
	if len(req.ParentA) > maxBaseLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "ParentA is too long"})
	}
	if len(req.ParentB) > maxBaseLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "ParentB is too long"})
	}
	if len(req.Content) > maxContentLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "Content is too long"})
	}
	if len(req.Endpoint) > maxBaseLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "Endpoint is too long"})
	}

	modelUpdatePayload := NewPayload(req.ModelID, req.ParentA, req.ParentB, req.Content, req.Endpoint)
	msg, err := deps.Tangle.IssuePayload(modelUpdatePayload)
	if err != nil {
		return c.JSON(http.StatusBadRequest, Response{Error: err.Error()})
	}

	return c.JSON(http.StatusOK, Response{MessageID: msg.ID().Base58()})
}

// Request defines the modelUpdate message to send.
type Request struct {
	ModelID  string `json:"modelID"`
	ParentA  string `json:"parentA"`
	ParentB  string `json:"parentB"`
	Content  string `json:"content"`
	Endpoint string `json:"endpoint"`
}

// Response contains the ID of the message sent.
type Response struct {
	MessageID string `json:"messageID,omitempty"`
	Error     string `json:"error,omitempty"`
}
