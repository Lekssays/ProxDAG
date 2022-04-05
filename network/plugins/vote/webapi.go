package vote

import (
	"net/http"

	"github.com/iotaledger/goshimmer/packages/jsonmodels"
	"github.com/labstack/echo"
)

const (
	maxBaseLength     = 512
	maxMetadataLength = 4096
)

func configureWebAPI() {
	deps.Server.POST("vote", SendVoteMessage)
}

// SendVoteMessage sends a vote message.
func SendVoteMessage(c echo.Context) error {
	req := &Request{}
	if err := c.Bind(req); err != nil {
		return c.JSON(http.StatusBadRequest, jsonmodels.NewErrorResponse(err))
	}

	if len(req.ModelID) > maxBaseLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "ModelID is too long"})
	}
	if len(req.Parents) > maxBaseLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "ElectionID is too long"})
	}

	if len(req.Content) > maxBaseLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "Decision is too long"})
	}
	if len(req.Endpoint) > maxMetadataLength {
		return c.JSON(http.StatusBadRequest, Response{Error: "Metadata is too long"})
	}

	votePayload := NewPayload(req.ModelID, req.Parents, req.Content, req.Endpoint)
	msg, err := deps.Tangle.IssuePayload(votePayload)
	if err != nil {
		return c.JSON(http.StatusBadRequest, Response{Error: err.Error()})
	}

	return c.JSON(http.StatusOK, Response{MessageID: msg.ID().Base58()})
}

// Request defines the vote message to send.
type Request struct {
	ModelID    string `json:"modelID"`
	ElectionID string `json:"electionID"`
	Decision   string `json:"decision"`
	Metadata   string `json:"metadata"`
}

// Response contains the ID of the message sent.
type Response struct {
	MessageID string `json:"messageID,omitempty"`
	Error     string `json:"error,omitempty"`
}
