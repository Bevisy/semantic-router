package gateway

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Router is the pipeline surface the HTTP gateway needs. *extproc.OpenAIRouter
// implements it via its GatewayProcessBody / GatewayProcessResponseBody /
// GatewayRelease methods; tests substitute fakes so the transport layer can
// be exercised without loading Rust binding dylibs.
type Router interface {
	// GatewayProcessBody runs the buffered request pipeline and returns the
	// HTTP-level outcome. A successful Forward carries a RequestID that the
	// caller passes back to GatewayProcessResponseBody for response-side
	// semantic processing, and to GatewayRelease for cleanup.
	GatewayProcessBody(ctx context.Context, sourcePath string, reqHeaders map[string]string, body []byte) *GatewayResult
	// GatewayProcessResponseBody runs the response-side semantic pipeline
	// (context recovery, jailbreak/hallucination filters, response logging,
	// caching) for one buffered response chunk, and returns the bytes the
	// caller should send the client. Streams call it per chunk with
	// endOfStream=false, then once more with endOfStream=true. Buffered
	// responses call it once with endOfStream=true.
	GatewayProcessResponseBody(requestID string, responseBody []byte, endOfStream bool) *GatewayResponseResult
	// GatewayRelease ends the request and releases its context (inflight
	// accounting, spans). Call it after the stream ends, after a buffered
	// response, or when the client disconnects early.
	GatewayRelease(requestID string)
}

// GatewayUpstream describes where the router decided to forward the request.
type GatewayUpstream struct {
	// BaseURL is the provider base (scheme + host + optional base path).
	BaseURL   string
	Path      string // request path (e.g. /v1/chat/completions), provider-adjusted
	WireBody  []byte // encoded upstream body (dispatch output)
	Format    llmprotocol.WireFormat
	Model     string // upstream model id
	AuthValue string // Authorization header value, if resolved
	Stream    bool
	// RequestID ties the forwarded request to its Router state for
	// response-side processing and release.
	RequestID string
}

// GatewayResult is the outcome of processing one request body.
type GatewayResult struct {
	// Immediate, when non-nil, means the router short-circuited (cache hit,
	// fast response, validation error, etc.) and the caller must reply with
	// this payload.
	Immediate *GatewayImmediate
	// Forward, when non-nil, means the request should be proxied upstream
	// with the given configuration.
	Forward *GatewayUpstream
	// Error carries a routing-level failure that should surface as an HTTP
	// error distinct from a crafted Immediate.
	Err error
}

// GatewayImmediate is a ready-to-send HTTP response (status + body).
type GatewayImmediate struct {
	Status  int
	Headers map[string]string
	Body    []byte
}

// GatewayResponseResult is the outcome of processing one response chunk.
type GatewayResponseResult struct {
	// Body is the (possibly rewritten) response bytes to send the client.
	// Nil means pass the original chunk through unchanged.
	Body []byte
	// Headers carries response-header mutations to apply (e.g. semantic
	// warning headers).
	Headers map[string]string
	// Immediate, when non-nil, means the response-side pipeline produced a
	// final HTTP response (e.g. an error reply) instead of a rewritten body.
	Immediate *GatewayImmediate
}
