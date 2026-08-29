package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExtractContentFast_FlowToolStateRequiresTrailingResult(t *testing.T) {
	flowBody := []byte(`{
		"model":"auto",
		"messages":[
			{"role":"user","content":"run the workflow"},
			{"role":"tool","tool_call_id":"flowtool_deadbeef__call_1","content":"done"}
		]
	}`)
	result, err := extractContentFast(flowBody)
	require.NoError(t, err)
	assert.True(t, result.LastMessageFlowToolResult)

	historicalBody := []byte(`{
		"model":"auto",
		"messages":[
			{"role":"tool","tool_call_id":"flowtool_deadbeef__call_1","content":"done"},
			{"role":"assistant","content":"workflow complete"},
			{"role":"user","content":"new request"}
		]
	}`)
	result, err = extractContentFast(historicalBody)
	require.NoError(t, err)
	assert.False(t, result.LastMessageFlowToolResult)
}

func TestExtractContentFast_HistoricalUnmatchedToolCallDoesNotStayActive(t *testing.T) {
	body := []byte(`{
		"model":"auto",
		"messages":[
			{"role":"assistant","content":null,"tool_calls":[{
				"id":"old-call","type":"function",
				"function":{"name":"lookup","arguments":"{}"}
			}]},
			{"role":"user","content":"start a separate task"},
			{"role":"assistant","content":"done"},
			{"role":"user","content":"ordinary follow-up"}
		]
	}`)
	r, err := extractContentFast(body)
	require.NoError(t, err)

	assert.Equal(t, 1, r.AssistantToolCallCount)
	assert.Zero(t, r.ToolResultCount)
	assert.False(t, r.LastAssistantToolCall)
	assert.False(t, r.LastMessageToolResult)
	assert.False(t, r.LastUserAfterToolResult)
}
