package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// handleResponseBody processes the response body.
func (r *OpenAIRouter) handleResponseBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	return r.processResponseBodyCore(v.ResponseBody.Body, v.ResponseBody.GetEndOfStream(), ctx)
}

// processResponseBodyCore runs the response-side semantic pipeline over a
// plain body, independent of any transport wire type. Both the Envoy ExtProc
// adapter (handleResponseBody) and the standalone HTTP gateway drive this
// same core so the res_filter pipeline (jailbreak/hallucination, usage,
// memory, caching) cannot diverge between the two binaries (#1138).
func (r *OpenAIRouter) processResponseBodyCore(
	responseBody []byte,
	endOfStream bool,
	ctx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if skipResponse := r.handleSkipProcessingResponseBody(responseBody, ctx); skipResponse != nil {
		return skipResponse, nil
	}

	completionLatency := time.Since(ctx.StartTime)

	// Decrement active request count for queue depth estimation.
	defer metrics.DecrementModelActiveRequests(ctx.RequestModel)

	if looperResponse := r.handleLooperResponseBody(responseBody, ctx); looperResponse != nil {
		return looperResponse, nil
	}

	if isUpstreamTransportError(ctx) {
		return r.handleUpstreamTransportError(responseBody, ctx), nil
	}

	if ctx.IsStreamingResponse {
		return r.handleSemanticStreamingResponseBody(responseBody, endOfStream, ctx), nil
	}
	recoveredBody, recoveryErr := r.handleContextRecoveryFollowup(
		ctx.TraceContext,
		responseBody,
		ctx,
	)
	if recoveryErr != nil {
		logging.ComponentWarnEvent("extproc", "context_recovery_followup_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      recoveryErr.Error(),
		})
		if contextRecoveryFailClosed(ctx) {
			return r.createErrorResponse(502, "Context recovery followup failed"), nil
		}
		responseBody = r.redactContextRecoveryToolCalls(responseBody, ctx)
	} else {
		responseBody = recoveredBody
	}

	return r.handleNonStreamingResponseBody(responseBody, ctx, completionLatency), nil
}

func contextRecoveryFailClosed(ctx *RequestContext) bool {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return false
	}
	plugin := ctx.VSRSelectedDecision.GetContextCompressionConfig()
	return plugin != nil &&
		plugin.EffectiveFailureMode() == config.ContextCompressionFailureClosed
}

func (r *OpenAIRouter) handleLooperResponseBody(
	responseBody []byte,
	ctx *RequestContext,
) *ext_proc.ProcessingResponse {
	if !ctx.LooperRequest {
		return nil
	}

	logging.Debugf("[Looper] Capturing response body for router replay")
	r.attachRouterReplayResponse(ctx, responseBody, true)
	return buildResponseBodyContinueResponse(nil, nil)
}

func buildResponseBodyContinueResponse(
	bodyMutation *ext_proc.BodyMutation,
	headerMutation *ext_proc.HeaderMutation,
) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ResponseBody{
			ResponseBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status:         ext_proc.CommonResponse_CONTINUE,
					HeaderMutation: headerMutation,
					BodyMutation:   bodyMutation,
				},
			},
		},
	}
}
