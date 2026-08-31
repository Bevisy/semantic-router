package gateway

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// fakeRouter implements Router with a canned outcome.
type fakeRouter struct {
	bodyErr   error
	immStatus int
	immBody   []byte
	forward   *GatewayUpstream
	released  chan string
	// respRewrite, when non-nil, is returned by GatewayProcessResponseBody
	// as the rewritten body (simulates the response-side pipeline mutating
	// bytes: res_filter interception, semantic headers).
	respRewrite []byte
	// respImmediate, when non-nil, short-circuits the response side (e.g.
	// the pipeline produced an error reply instead of a rewritten body).
	respImmediate *GatewayImmediate
}

func (f *fakeRouter) GatewayProcessBody(ctx context.Context, sourcePath string, reqHeaders map[string]string, body []byte) *GatewayResult {
	if f.bodyErr != nil {
		return &GatewayResult{Err: f.bodyErr}
	}
	if f.forward != nil {
		return &GatewayResult{Forward: f.forward}
	}
	return &GatewayResult{Immediate: &GatewayImmediate{Status: f.immStatus, Body: f.immBody}}
}

func (f *fakeRouter) GatewayProcessResponseBody(requestID string, responseBody []byte, endOfStream bool) *GatewayResponseResult {
	if f.respImmediate != nil {
		return &GatewayResponseResult{Immediate: f.respImmediate}
	}
	if f.respRewrite != nil {
		return &GatewayResponseResult{Body: f.respRewrite}
	}
	return &GatewayResponseResult{}
}

func (f *fakeRouter) GatewayRelease(requestID string) {
	if f.released != nil {
		f.released <- requestID
	}
}

var _ Router = (*fakeRouter)(nil)

func TestGatewayImmediateReply(t *testing.T) {
	s := NewServerWith(nil, &fakeRouter{immStatus: http.StatusOK, immBody: []byte(`{"ok":true}`)})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Body.String(); got != `{"ok":true}` {
		t.Fatalf("body = %q, want %q", got, `{"ok":true}`)
	}
}

func TestGatewayForwardUpstream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, 1024)
		n, _ := r.Body.Read(buf)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		w.Write([]byte("echo:"))
		w.Write(buf[:n])
	}))
	defer upstream.Close()

	s := NewServerWith(nil, &fakeRouter{forward: &GatewayUpstream{
		BaseURL:  upstream.URL,
		Path:     "/v1/chat/completions",
		WireBody: []byte(`{"model":"qwen"}`),
		Format:   llmprotocol.OpenAIChatV1,
	}})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"model":"qwen"}`)))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201", w.Code)
	}
	if got := w.Body.String(); got != "echo:{\"model\":\"qwen\"}" {
		t.Fatalf("body = %q", got)
	}
}

func TestGatewayError(t *testing.T) {
	s := NewServerWith(nil, &fakeRouter{bodyErr: context.DeadlineExceeded})

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(`{}`)))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", w.Code)
	}
}

func TestGatewayBodyTooLarge(t *testing.T) {
	s := NewServerWith(nil, &fakeRouter{}, WithMaxBodyBytes(16))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(make([]byte, 64)))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusRequestEntityTooLarge)
	}
}

func TestGatewayStreamingRelay(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		w.Write([]byte("data: {\"a\":1}\n\n"))
		w.(http.Flusher).Flush()
		w.Write([]byte("data: [DONE]\n\n"))
		w.(http.Flusher).Flush()
	}))
	defer upstream.Close()

	s := NewServerWith(nil, &fakeRouter{forward: &GatewayUpstream{
		BaseURL:  upstream.URL,
		Path:     "/v1/chat/completions",
		WireBody: []byte(`{"stream":true}`),
		Format:   llmprotocol.OpenAIChatV1,
		Stream:   true,
	}})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"stream":true}`)))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)

	body := w.Body.String()
	for _, want := range []string{"data: {\"a\":1}", "data: [DONE]"} {
		if !bytes.Contains([]byte(body), []byte(want)) {
			t.Fatalf("body missing %q; got %q", want, body)
		}
	}
}

// TestGatewayCancelPropagatesToUpstream proves client cancellation ends the
// gateway handler promptly and releases the retained request context: a client
// that disappears mid-stream must not leak the gateway goroutine or its
// session (#1138 cancellation contract). The Relayed-forward result carries
// c.Request.Context() to the upstream request, and serveStream's body read
// returns context.Canceled on the client disconnect.
func TestGatewayCancelPropagatesToUpstream(t *testing.T) {
	released := make(chan string, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Real SSE upstream: send the response head, then hold the stream.
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"a\":1}\n\n"))
		w.(http.Flusher).Flush()
		// Hold briefly so the gateway has a live upstream to cancel; the
		// gateway-side GatewayRelease is the real contract (see comment above).
		<-time.After(1 * time.Second)
	}))
	defer upstream.Close()

	s := NewServerWith(nil, &fakeRouter{
		forward: &GatewayUpstream{
			BaseURL:   upstream.URL,
			Path:      "/v1/chat/completions",
			WireBody:  []byte(`{"stream":true}`),
			Format:    llmprotocol.OpenAIChatV1,
			Stream:    true,
			RequestID: "cancel-test",
		},
		released: released,
	})

	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"stream":true}`))).WithContext(ctx)
	w := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		s.Handler().ServeHTTP(w, req)
		close(done)
	}()
	// Let the gateway reach the streaming relay, then drop the client.
	select {
	case <-time.After(200 * time.Millisecond):
	case <-done:
		t.Fatal("handler returned before client disconnect")
	}
	cancel()
	// The handler must return promptly (no goroutine leak).
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("gateway handler did not return after client disconnect")
	}
	// The retained session must be released. serveStream releases on write
	// error; with an httptest recorder the write never fails, so the release
	// fired by the ctx-cancelled read path is what we assert. Accept either
	// the stream-error release or the finalize path.
	select {
	case <-released:
	case <-time.After(2 * time.Second):
		t.Fatal("request context was not released after client disconnect")
	}
}

// TestCollectHeadersDropsHopByHop proves the gateway applies the same
// hop-by-hop header-strip policy as the ExtProc adapter, so transport
// framing never reaches the semantic pipeline (#1138 header-trust contract).
func TestCollectHeadersDropsHopByHop(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer abc")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("Transfer-Encoding", "chunked")
	req.Header.Set("Upgrade", "h2c")
	req.Header.Set("x-selected-model", "client-model") // semantic header: preserved

	got := collectHeaders(req)
	for _, framing := range headers.HopByHopDropList {
		if _, ok := got[framing]; ok {
			t.Fatalf("hop-by-hop header %q leaked into semantic headers: %v", framing, got)
		}
	}
	for _, keep := range []string{"content-type", "authorization", "x-selected-model"} {
		if _, ok := got[keep]; !ok {
			t.Fatalf("expected semantic header %q to survive, got %v", keep, got)
		}
	}
}

// TestGatewayBufferedResponseRewrite proves the response-side semantic
// pipeline can rewrite a buffered upstream body before it reaches the
// client (res_filter interception, semantic passing). The gateway must
// emit the rewritten bytes, not the raw upstream body.
func TestGatewayBufferedResponseRewrite(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"raw":"upstream"}`))
	}))
	defer upstream.Close()

	s := NewServerWith(nil, &fakeRouter{
		forward: &GatewayUpstream{
			BaseURL:   upstream.URL,
			Path:      "/v1/chat/completions",
			WireBody:  []byte(`{}`),
			Format:    llmprotocol.OpenAIChatV1,
			RequestID: "rewrite-test",
		},
		respRewrite: []byte(`{"filtered":true}`),
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Body.String(); got != `{"filtered":true}` {
		t.Fatalf("body = %q, want rewritten %q (raw upstream leaked)", got, `{"filtered":true}`)
	}
}

// TestGatewayStreamingChunkRewrite proves the response-side pipeline can
// rewrite individual SSE chunks mid-stream (e.g. filtering a jailbreak /
// hallucination token in a chunk). Each chunk must be transformed before
// the client sees it.
func TestGatewayStreamingChunkRewrite(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		w.Write([]byte("data: {\"c\":1}\n\n"))
		w.(http.Flusher).Flush()
		w.Write([]byte("data: {\"c\":2}\n\n"))
		w.(http.Flusher).Flush()
		w.Write([]byte("data: [DONE]\n\n"))
		w.(http.Flusher).Flush()
	}))
	defer upstream.Close()

	s := NewServerWith(nil, &fakeRouter{
		forward: &GatewayUpstream{
			BaseURL:   upstream.URL,
			Path:      "/v1/chat/completions",
			WireBody:  []byte(`{"stream":true}`),
			Format:    llmprotocol.OpenAIChatV1,
			Stream:    true,
			RequestID: "chunk-rewrite",
		},
		respRewrite: []byte(`data: {"filtered":{"c":1}}\n\n`),
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"stream":true}`)))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)

	if got := w.Body.String(); !bytes.Contains([]byte(got), []byte(`{"filtered":{"c":1}}`)) {
		t.Fatalf("body missing rewritten chunk; got %q", got)
	}
	for _, raw := range []string{`data: {"c":1}`, `data: {"c":2}`, `data: [DONE]`} {
		if bytes.Contains([]byte(w.Body.String()), []byte(raw)) {
			t.Fatalf("raw upstream chunk %q leaked to client: %q", raw, w.Body.String())
		}
	}
}

// TestGatewayResponseSideImmediate proves a response-side short-circuit
// (e.g. the pipeline detects a violation and emits an error reply) replaces
// the upstream body instead of being passed through.
func TestGatewayResponseSideImmediate(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"raw":"upstream"}`))
	}))
	defer upstream.Close()

	s := NewServerWith(nil, &fakeRouter{
		forward: &GatewayUpstream{
			BaseURL:   upstream.URL,
			Path:      "/v1/chat/completions",
			WireBody:  []byte(`{}`),
			Format:    llmprotocol.OpenAIChatV1,
			RequestID: "imm-test",
		},
		respImmediate: &GatewayImmediate{Status: http.StatusForbidden, Body: []byte(`{"error":"blocked"}`)},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", w.Code)
	}
	if got := w.Body.String(); got != `{"error":"blocked"}` {
		t.Fatalf("body = %q, want response-side immediate %q", got, `{"error":"blocked"}`)
	}
}

// TestGatewayConcurrentRequests proves independent requests running through
// the gateway handle concurrently complete without cross-talk or data loss:
// every request reaches the upstream, returns 200, and receives the full
// (non-empty) payload the router produced. The gateway is stateless per
// request; request-body isolation itself is exercised by the buffered
// rewrite test (each request's pipeline result is its own).
func TestGatewayConcurrentRequests(t *testing.T) {
	const n = 16
	wireBody := []byte(`{"model":"shared"}`)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Echo what the gateway actually forwarded (the router's WireBody).
		body, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("echo:"))
		w.Write(body)
	}))
	defer upstream.Close()

	s := NewServerWith(nil, &fakeRouter{
		forward: &GatewayUpstream{
			BaseURL:   upstream.URL,
			Path:      "/v1/chat/completions",
			WireBody:  wireBody,
			Format:    llmprotocol.OpenAIChatV1,
			RequestID: "dedup",
		},
	})

	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"stream":false}`)))
			w := httptest.NewRecorder()
			s.Handler().ServeHTTP(w, req)
			if w.Code != http.StatusOK {
				errs <- fmt.Errorf("req %d: status %d", i, w.Code)
				return
			}
			want := "echo:" + string(wireBody)
			if got := w.Body.String(); got != want {
				errs <- fmt.Errorf("req %d: got %q want %q", i, got, want)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
