package extproc

// Neutral gateway pipeline surface. The gateway transport layer (pkg/gateway)
// depends only on these methods and the neutral result types in pkg/gateway;
// no Envoy ExtProc type crosses this boundary (see #1138 acceptance criterion
// 1: shared routing input/output contracts carry no Envoy protobuf
// dependency). Internally this file adapts to the same handleRequestBody /
// handleResponseBody pipeline the Envoy ExtProc adapter drives, so gateway
// and Envoy paths cannot drift.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/inflight"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// gatewaySessions retains the per-request RequestContext between the request
// (GatewayProcessBody) and response (GatewayProcessResponseBody) phases so the
// response-side semantic pipeline (res_filter, streaming accumulation,
// response logging/caching) runs on the same request state the ExtProc path
// uses. Entries are released via GatewayRelease when a request completes or
// the client disconnects.
// ponytail: single package-level map is fine for a single-gateway process;
// shard per-request if a future process serves many concurrent streams.
var gatewaySessions sync.Map // requestID -> *RequestContext

// GatewayProcessBody runs the request pipeline for one buffered body and
// returns the HTTP-level outcome. sourcePath is the public request path used
// for source-format detection.
func (r *OpenAIRouter) GatewayProcessBody(
	ctx context.Context,
	sourcePath string,
	reqHeaders map[string]string,
	body []byte,
) *gateway.GatewayResult {
	if r == nil {
		return &gateway.GatewayResult{Err: errors.New("router is unavailable")}
	}
	rcx := &RequestContext{
		Headers:      reqHeaders,
		StartTime:    time.Now(),
		TraceContext: ctx,
	}
	if rcx.Headers == nil {
		rcx.Headers = make(map[string]string)
	}
	rcx.RequestID = rcx.Headers["x-vsr-request-id"]
	if rcx.RequestID == "" {
		rcx.RequestID = newGatewayRequestID()
	}

	// Same source-format detection the header phase performs.
	detectSourceFormat(sourcePath, rcx)
	detectStreamingExpectation(rcx)

	resp, err := r.processRequestBodyCore(body, rcx)
	if err != nil {
		// Entrypoint/model routing failures are usually already converted to
		// crafted responses; only structural errors land here.
		r.finalizeGatewayContext(rcx)
		return &gateway.GatewayResult{Err: fmt.Errorf("gateway body processing: %w", err)}
	}

	result := r.translateGatewayResponse(resp, rcx)
	if result.Forward != nil {
		// Keep the request context alive for the response phase (res_filter +
		// streaming accumulation run on it). The server calls GatewayRelease
		// when the response completes or the client disconnects.
		// No response-header phase runs in the gateway, so derive the
		// streaming flag here the way evaluateResponseHeaderOutcome would:
		// stream requests must route response chunks through the semantic
		// streaming pipeline, not the buffered JSON decode.
		rcx.IsStreamingResponse = rcx.ExpectStreamingResponse
		result.Forward.RequestID = rcx.RequestID
		gatewaySessions.Store(rcx.RequestID, rcx)
	} else {
		// Immediate/error replies end the request here; nothing to retain.
		r.finalizeGatewayContext(rcx)
	}
	return result
}

func (r *OpenAIRouter) finalizeGatewayContext(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	if ctx.InflightToken != 0 {
		inflight.End(ctx.RequestModel, ctx.InflightToken)
		ctx.InflightToken = 0
	}
	if ctx.UpstreamSpan != nil {
		tracing.RecordError(ctx.UpstreamSpan, nil) // no-op unless an error was set
		ctx.UpstreamSpan.End()
	}
}

// translateGatewayResponse converts a pipeline wire response into the
// HTTP-facing outcome.
func (r *OpenAIRouter) translateGatewayResponse(resp *ext_proc.ProcessingResponse, ctx *RequestContext) *gateway.GatewayResult {
	if resp == nil {
		return &gateway.GatewayResult{Err: errors.New("empty pipeline response")}
	}
	if imm := resp.GetImmediateResponse(); imm != nil {
		return &gateway.GatewayResult{Immediate: gatewayImmediate(imm)}
	}
	if rb := resp.GetRequestBody(); rb != nil {
		// CommonResponse with status CONTINUE + header/body mutation is the
		// "forward upstream with these modifications" contract. If the
		// pipeline wants additional body handling (e.g. semantic plugins),
		// the mutation carries the final upstream body.
		common := rb.GetResponse()
		if common == nil {
			return &gateway.GatewayResult{Err: errors.New("body response without common response")}
		}
		if common.Status != ext_proc.CommonResponse_CONTINUE {
			return &gateway.GatewayResult{Err: fmt.Errorf("unexpected body response status %s", common.Status.String())}
		}
		upstream := r.gatewayUpstreamFromMutation(common, ctx)
		if upstream == nil {
			return &gateway.GatewayResult{Err: errors.New("provider dispatch produced no upstream target")}
		}
		return &gateway.GatewayResult{Forward: upstream}
	}
	return &gateway.GatewayResult{Err: fmt.Errorf("unsupported pipeline response kind: %T", resp.Response)}
}

func gatewayImmediate(imm *ext_proc.ImmediateResponse) *gateway.GatewayImmediate {
	status := http.StatusInternalServerError
	hdrs := make(map[string]string)
	if imm.Status != nil {
		status = int(imm.Status.Code)
	}
	for _, h := range imm.Headers.GetSetHeaders() {
		key := strings.ToLower(h.GetHeader().GetKey())
		hdrs[key] = string(h.GetHeader().GetRawValue())
	}
	body := imm.GetBody()
	if body == nil {
		return &gateway.GatewayImmediate{Status: status, Headers: hdrs}
	}
	return &gateway.GatewayImmediate{Status: status, Headers: hdrs, Body: body}
}

// gatewayUpstreamFromMutation reconstructs the upstream request from the
// CONTINUE mutation the pipeline produced. The mutation carries provider
// credentials, profile headers, and the provider-adjusted :path; the body and
// format come from the request context.
func (r *OpenAIRouter) gatewayUpstreamFromMutation(common *ext_proc.CommonResponse, ctx *RequestContext) *gateway.GatewayUpstream {
	if ctx == nil || ctx.SemanticRequest == nil {
		return nil
	}
	gw := &gateway.GatewayUpstream{
		Format:   ctx.TargetFormat,
		Model:    ctx.SemanticRequest.Model,
		WireBody: nil, // filled below
		Stream:   ctx.SemanticRequest.Stream,
	}
	if gw.Format == "" {
		gw.Format = ctx.SourceFormat
	}
	if gw.Format == "" {
		gw.Format = "openai.chat.v1"
	}

	// Resolve base URL from the endpoint the router selected. The logical
	// model (the config key, e.g. "qwen/general") is written by the pipeline
	// into the x-selected-model mutation header; ctx.SemanticRequest.Model is
	// only the upstream model id and cannot be used to re-resolve the backend.
	logicalModel := ""
	for _, h := range common.GetHeaderMutation().GetSetHeaders() {
		key := h.GetHeader().GetKey()
		value := string(h.GetHeader().GetRawValue())
		switch key {
		case ":path":
			gw.Path = value
		case "authorization", "Authorization":
			gw.AuthValue = value
		case headers.SelectedModel:
			logicalModel = value
		}
	}
	if logicalModel == "" {
		logicalModel = gw.Model
	}
	backendAddress, _, found, err := r.Config.ResolvePrimaryBackendForModel(logicalModel)
	if err != nil || !found {
		logging.ComponentErrorEvent("gateway", "backend_resolve_failed", map[string]interface{}{
			"request_id": ctx.RequestID, "model": logicalModel, "error": errValue(err),
		})
		return nil
	}
	gw.BaseURL = backendAddress

	// The upstream body is what the pipeline encoded for dispatch. If a body
	// mutation is present it supersedes (semantic plugins rewrote it).
	if bm := common.GetBodyMutation(); bm != nil {
		if body := bm.GetBody(); len(body) > 0 {
			gw.WireBody = body
		}
	}
	if gw.WireBody == nil {
		encoded, err := r.encodeDispatchRequest(ctx)
		if err != nil {
			logging.ComponentErrorEvent("gateway", "encode_dispatch_failed", map[string]interface{}{
				"request_id": ctx.RequestID, "error": errValue(err),
			})
			return nil
		}
		gw.WireBody = encoded
	}
	if gw.Path == "" {
		gw.Path = requestWirePath(gw.Format)
	}
	return gw
}

func errValue(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// newGatewayRequestID returns a random hex request id used when the caller
// did not supply one through headers.
func newGatewayRequestID() string {
	var buf [8]byte
	if _, err := rand.Read(buf[:]); err == nil {
		return hex.EncodeToString(buf[:])
	}
	return fmt.Sprintf("gw-%d", time.Now().UnixNano())
}

// GatewayProcessResponseBody runs the response-side semantic pipeline for one
// response chunk, on the same RequestContext retained by GatewayProcessBody.
// Streaming callers invoke it per chunk with endOfStream=false and then once
// with endOfStream=true; buffered callers invoke it once with endOfStream=true.
// The returned Body is nil when the chunk passes through unchanged.
func (r *OpenAIRouter) GatewayProcessResponseBody(
	requestID string,
	responseBody []byte,
	endOfStream bool,
) *gateway.GatewayResponseResult {
	if r == nil {
		return &gateway.GatewayResponseResult{Immediate: &gateway.GatewayImmediate{Status: http.StatusInternalServerError, Body: []byte("router is unavailable")}}
	}
	anyCtx, ok := gatewaySessions.Load(requestID)
	if !ok || anyCtx == nil {
		// No retained request context: nothing to run, pass the bytes through
		// (the request was probably already released or never retained).
		return &gateway.GatewayResponseResult{}
	}
	ctx, ok := anyCtx.(*RequestContext)
	if !ok || ctx == nil {
		gatewaySessions.Delete(requestID)
		return &gateway.GatewayResponseResult{}
	}

	resp, err := r.processResponseBodyCore(responseBody, endOfStream, ctx)
	if err != nil {
		// Structural response-processing failure; surface as a 502 for the
		// last chunk (streaming) or the buffered reply.
		if endOfStream {
			r.GatewayRelease(requestID)
		}
		return &gateway.GatewayResponseResult{Immediate: &gateway.GatewayImmediate{Status: http.StatusBadGateway, Body: []byte("response processing failed")}}
	}
	result := translateGatewayResponseToHTTP(resp)
	if endOfStream {
		r.GatewayRelease(requestID)
	}
	return result
}

// GatewayRelease ends a gateway request and releases its retained context,
// flushing inflight accounting and spans. Safe to call twice.
func (r *OpenAIRouter) GatewayRelease(requestID string) {
	anyCtx, ok := gatewaySessions.LoadAndDelete(requestID)
	if !ok || anyCtx == nil {
		return
	}
	if ctx, ok := anyCtx.(*RequestContext); ok {
		r.finalizeGatewayContext(ctx)
	}
}

// translateGatewayResponseToHTTP converts a response-side wire reply into the
// HTTP-facing outcome: an Immediate (error/warning reply) or a rewritten body
// with optional header mutations. A nil Body means pass through unchanged.
func translateGatewayResponseToHTTP(resp *ext_proc.ProcessingResponse) *gateway.GatewayResponseResult {
	if resp == nil {
		return &gateway.GatewayResponseResult{}
	}
	if imm := resp.GetImmediateResponse(); imm != nil {
		return &gateway.GatewayResponseResult{Immediate: gatewayImmediate(imm)}
	}
	if rb := resp.GetResponseBody(); rb != nil {
		common := rb.GetResponse()
		if common == nil {
			return &gateway.GatewayResponseResult{}
		}
		hdrs := make(map[string]string)
		for _, h := range common.GetHeaderMutation().GetSetHeaders() {
			key := strings.ToLower(h.GetHeader().GetKey())
			hdrs[key] = string(h.GetHeader().GetRawValue())
		}
		var body []byte
		if bm := common.GetBodyMutation(); bm != nil {
			body = bm.GetBody()
		}
		return &gateway.GatewayResponseResult{Body: body, Headers: hdrs}
	}
	return &gateway.GatewayResponseResult{}
}
