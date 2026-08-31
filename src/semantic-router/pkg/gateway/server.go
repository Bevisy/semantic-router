// Package gateway exposes the semantic router as a standalone HTTP gateway,
// mirroring the ExtProc pipeline over plain OpenAI/Anthropic/Responses HTTP
// endpoints. The request pipeline (routing, plugins, translation) runs in
// exactly the same code path used by Envoy ExtProc; this package only replaces
// the transport boundary.
package gateway

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Server is the standalone HTTP gateway.
type Server struct {
	router       Router
	client       *http.Client
	addr         string
	timeout      time.Duration
	maxBodyBytes int64
}

// Option configures a gateway Server.
type Option func(*Server)

// WithTimeout sets the upstream request timeout (0 = http.DefaultClient's).
func WithTimeout(d time.Duration) Option { return func(s *Server) { s.timeout = d } }

// WithMaxBodyBytes bounds the request body read at the transport edge. It
// default aligns with the router pipeline's BodyBytes limit so oversized
// bodies are rejected before they are fully buffered (see llmprotocol policy).
func WithMaxBodyBytes(n int64) Option { return func(s *Server) { s.maxBodyBytes = n } }

// NewServer constructs a gateway server around an existing router.
func NewServer(router Router, opts ...Option) *Server {
	s := &Server{router: router, addr: ":8080", maxBodyBytes: int64(llmprotocol.DefaultPolicy().Limits.BodyBytes)}
	// DisableKeepAlives so upstream connections are closed when a request
	// context is cancelled: with keep-alive reuse, cancelling one request
	// returns the connection to the idle pool instead of severing it, and
	// an upstream blocked on the wire would never observe the disconnect
	// (#1138 cancellation contract).
	s.client = &http.Client{Transport: &http.Transport{DisableKeepAlives: true}}
	for _, o := range opts {
		o(s)
	}
	if s.maxBodyBytes <= 0 {
		s.maxBodyBytes = int64(llmprotocol.DefaultPolicy().Limits.BodyBytes)
	}
	if s.timeout > 0 {
		s.client.Timeout = s.timeout
	}
	return s
}

// NewServerWith constructs a gateway server with an explicit router gateway
// implementation (used by tests to inject fakes; callers normally use NewServer).
func NewServerWith(router Router, gw Router, opts ...Option) *Server {
	s := NewServer(router, opts...)
	if gw != nil {
		s.router = gw
	}
	return s
}

// Handler returns the routing engine that serves the three inference formats.
func (s *Server) Handler() http.Handler {
	r := gin.New()
	r.Use(gin.Recovery())
	// OpenAI Chat Completions, Anthropic Messages, OpenAI Responses.
	r.POST("/v1/chat/completions", s.handleInference("/v1/chat/completions"))
	r.POST("/v1/messages", s.handleInference("/v1/messages"))
	r.POST("/v1/responses", s.handleInference("/v1/responses"))
	r.GET("/v1/models", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"object": "list", "data": []string{}}) })
	return r
}

// ListenAndServe starts the gateway server on s.addr.
func (s *Server) ListenAndServe() error {
	logging.Infof("gateway listening on %s", s.addr)
	return http.ListenAndServe(s.addr, s.Handler())
}

// ListenAndServeAddr starts the gateway server on the given address.
func (s *Server) ListenAndServeAddr(addr string) error {
	logging.Infof("gateway listening on %s", addr)
	return http.ListenAndServe(addr, s.Handler())
}

func (s *Server) handleInference(sourcePath string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Bound the body read at the transport edge (#1138 body-limits
		// alignment): reject oversized bodies before buffering them whole.
		if c.Request.Body != nil {
			c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, s.maxBodyBytes)
		}
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			logging.ComponentWarnEvent("gateway", "request_body_rejected", map[string]interface{}{
				"path": sourcePath, "limit_bytes": s.maxBodyBytes, "error": err.Error(),
			})
			c.Data(http.StatusRequestEntityTooLarge, "application/json", []byte(`{"error":{"message":"request body is empty or exceeds the configured limit"}}`))
			return
		}
		defer c.Request.Body.Close()
		reqHeaders := collectHeaders(c.Request)
		start := time.Now()
		result := s.router.GatewayProcessBody(c.Request.Context(), sourcePath, reqHeaders, body)

		if result.Err != nil {
			logging.Errorf("gateway processing error: %s (path %s)", result.Err.Error(), sourcePath)
			writeError(c, http.StatusInternalServerError, result.Err.Error())
			return
		}
		if result.Immediate != nil {
			writeImmediate(c, result.Immediate)
			return
		}
		if result.Forward != nil {
			s.forward(c, result.Forward, sourcePath)
			return
		}
		writeError(c, http.StatusInternalServerError, "unreachable: no pipeline outcome")
		_ = start
	}
}

// collectHeaders flattens the incoming request headers into the map the
// router's header phases expect. Keys are lower-cased so pipeline lookups
// (which read e.g. "accept") work regardless of the client's casing.
// Transport framing headers are dropped (shared HopByHopDropList, same as
// the ExtProc adapter) so header trust is consistent across adapters.
func collectHeaders(r *http.Request) map[string]string {
	out := make(map[string]string, len(r.Header)+4)
	for k, vs := range r.Header {
		if len(vs) == 0 {
			continue
		}
		lk := strings.ToLower(k)
		if slices.Contains(headers.HopByHopDropList, lk) {
			continue
		}
		out[lk] = strings.Join(vs, ",")
	}
	// Envoy-style pseudo headers used by the request-header stage.
	out[":method"] = r.Method
	out[":path"] = r.URL.Path
	if r.URL.RawQuery != "" {
		out[":path"] = r.URL.Path + "?" + r.URL.RawQuery
	}
	if host := r.Host; host != "" {
		out[":authority"] = host
	}
	return out
}

func writeImmediate(c *gin.Context, imm *GatewayImmediate) {
	status := imm.Status
	if status == 0 {
		status = http.StatusOK
	}
	ct := "application/json"
	for k, v := range imm.Headers {
		if strings.EqualFold(k, "content-type") {
			ct = v
			continue
		}
		c.Header(k, v)
	}
	if len(imm.Body) == 0 {
		c.Status(status)
		return
	}
	c.Data(status, ct, imm.Body)
}

func writeError(c *gin.Context, status int, message string) {
	payload := fmt.Sprintf(`{"error":{"message":%q}}`, message)
	c.Data(status, "application/json", []byte(payload))
}

// forward proxies the request to the upstream the router selected.
func (s *Server) forward(c *gin.Context, up *GatewayUpstream, sourcePath string) {
	target := up.BaseURL + up.Path
	// Backend endpoints may be configured without a scheme (host:port) in
	// Envoy mode, where the cluster resolves the transport. The HTTP gateway
	// must supply one.
	if !strings.Contains(target, "://") {
		target = "http://" + target
	}
	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, target, bytes.NewReader(up.WireBody))
	if err != nil {
		writeError(c, http.StatusBadGateway, "failed to build upstream request")
		return
	}
	// Preserve the client content-type, override anything upstream needs.
	ct := c.GetHeader("Content-Type")
	if ct == "" {
		ct = "application/json"
	}
	req.Header.Set("Content-Type", ct)
	if up.Stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	if up.AuthValue != "" {
		req.Header.Set("Authorization", up.AuthValue)
	}
	if rID := c.GetHeader("x-request-id"); rID != "" {
		req.Header.Set("x-request-id", rID)
	}

	upstreamStart := time.Now()
	resp, err := s.client.Do(req)
	if err != nil {
		logging.Errorf("gateway upstream error: %v (target %s)", err, target)
		// Tolerate context cancellation (client disconnected).
		if errors.Is(err, context.Canceled) {
			return
		}
		writeError(c, http.StatusBadGateway, fmt.Sprintf("upstream request failed: %v", err))
		return
	}
	logging.ComponentEvent("gateway", "upstream_response_received", map[string]interface{}{
		"request_id": up.RequestID, "target": target, "status": resp.StatusCode,
		"ttfb_ms": time.Since(upstreamStart).Milliseconds(),
	})
	defer resp.Body.Close()

	// Copy response headers the caller should see. Length headers are
	// dropped: the semantic pipeline below may rewrite the body, and Go's
	// writer recomputes Content-Length from what is actually sent.
	for k, vs := range resp.Header {
		if strings.EqualFold(k, "Content-Length") || strings.EqualFold(k, "Transfer-Encoding") {
			continue
		}
		for _, v := range vs {
			c.Header(k, v)
		}
	}

	if up.Stream {
		s.serveStream(c, resp, up.RequestID)
		return
	}
	// Buffered response: run the same response-side semantic pipeline the
	// ExtProc path runs (res_filter hallucinations/jailbreak, response
	// logging, caching), then emit the possibly-rewritten body.
	respBody, rerr := io.ReadAll(resp.Body)
	if rerr != nil {
		writeError(c, http.StatusBadGateway, "failed to read upstream response")
		return
	}
	if up.RequestID != "" {
		rresult := s.router.GatewayProcessResponseBody(up.RequestID, respBody, true)
		if rresult.Immediate != nil {
			writeImmediate(c, rresult.Immediate)
			return
		}
		for k, v := range rresult.Headers {
			c.Header(k, v)
		}
		if rresult.Body != nil {
			respBody = rresult.Body
		}
	}
	status := resp.StatusCode
	if status == 0 {
		status = http.StatusOK
	}
	c.Data(status, resp.Header.Get("Content-Type"), respBody)
}

// serveStream relays an upstream SSE stream chunk-by-chunk, running each
// chunk through the response-side semantic pipeline before flushing, and
// cancelling when the client disconnects. The final (end-of-stream) chunk
// releases the retained request context.
func (s *Server) serveStream(c *gin.Context, resp *http.Response, requestID string) {
	c.Status(resp.StatusCode)
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		c.Header("Content-Type", ct)
	}
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	buf := make([]byte, 32*1024)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			chunk := buf[:n]
			body := s.processStreamChunk(requestID, chunk, false)
			if body != nil {
				chunk = body
			}
			if _, werr := c.Writer.Write(chunk); werr != nil {
				if requestID != "" {
					s.router.GatewayRelease(requestID)
				}
				return // client gone; upstream request ctx cancels via c.Request.Context()
			}
			c.Writer.Flush()
		}
		if err != nil {
			if err != io.EOF {
				logging.Errorf("gateway stream error: %v", err)
				if requestID != "" {
					s.router.GatewayRelease(requestID)
				}
				return
			}
			break
		}
	}
	// End of stream: run the semantic finalize (needed for usage/cache
	// accounting even when the stream is a pass-through).
	if requestID != "" {
		final := s.router.GatewayProcessResponseBody(requestID, nil, true)
		if final.Immediate != nil {
			writeImmediate(c, final.Immediate)
			return
		}
		if final.Body != nil {
			if _, werr := c.Writer.Write(final.Body); werr == nil {
				c.Writer.Flush()
			}
		}
	}
}

// processStreamChunk runs one non-terminal stream chunk through the response
// semantic pipeline ordered the way the ExtProc path does. It returns the
// rewritten bytes when the pipeline produced a body mutation, or nil for a
// pass-through.
func (s *Server) processStreamChunk(requestID string, chunk []byte, eos bool) []byte {
	if requestID == "" {
		return nil
	}
	rresult := s.router.GatewayProcessResponseBody(requestID, chunk, eos)
	if rresult == nil {
		return nil
	}
	if rresult.Immediate != nil {
		return rresult.Immediate.Body
	}
	return rresult.Body
}
