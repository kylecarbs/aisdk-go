package aisdk_test

import (
	"testing"

	"github.com/kylecarbs/aisdk-go"
	"github.com/stretchr/testify/require"
)

func TestDataStreamAccumulator_SimpleText(t *testing.T) {
	t.Parallel()

	parts := []aisdk.DataStreamPart{
		aisdk.StartStepStreamPart{MessageID: "msg_01E71z89cub9AaDBUreABi5E"},
		aisdk.TextStreamPart{Content: "I'd be happy to"},
		aisdk.TextStreamPart{Content: " chat with you! Is there something specific you'd like"},
		aisdk.TextStreamPart{Content: " to talk about? I can discuss a wide range of topics including"},
		aisdk.TextStreamPart{Content: ":\n\n- Current events\n- Technology\n- Science\n-"},
		aisdk.TextStreamPart{Content: " Arts and entertainment\n- Philosophy\n- Personal interests"},
		aisdk.TextStreamPart{Content: " or hobbies\n- Advice"},
		aisdk.TextStreamPart{Content: " on various subjects\n\nFeel free to let"},
		aisdk.TextStreamPart{Content: " me know what's on your mind, and we"},
		aisdk.TextStreamPart{Content: " can have a conversation about it."},
		aisdk.FinishMessageStreamPart{
			FinishReason: aisdk.FinishReasonStop,
			Usage: aisdk.Usage{
				PromptTokens:     int64Ptr(10),
				CompletionTokens: int64Ptr(90),
			},
		},
	}

	expectedMessage := aisdk.Message{
		ID:   "msg_01E71z89cub9AaDBUreABi5E",
		Role: "assistant",
		Content: "I'd be happy to chat with you! Is there something specific you'd like" +
			" to talk about? I can discuss a wide range of topics including" +
			":\n\n- Current events\n- Technology\n- Science\n-" +
			" Arts and entertainment\n- Philosophy\n- Personal interests" +
			" or hobbies\n- Advice" +
			" on various subjects\n\nFeel free to let" +
			" me know what's on your mind, and we" +
			" can have a conversation about it.",
		Parts: []aisdk.Part{
			{
				Type: aisdk.PartTypeStepStart,
			},
			{
				Type: aisdk.PartTypeText,
				Text: "I'd be happy to chat with you! Is there something specific you'd like" +
					" to talk about? I can discuss a wide range of topics including" +
					":\n\n- Current events\n- Technology\n- Science\n-" +
					" Arts and entertainment\n- Philosophy\n- Personal interests" +
					" or hobbies\n- Advice" +
					" on various subjects\n\nFeel free to let" +
					" me know what's on your mind, and we" +
					" can have a conversation about it.",
			},
		},
	}

	var acc aisdk.DataStreamAccumulator
	for _, part := range parts {
		err := acc.Push(part)
		if err != nil {
			t.Fatalf("acc.Push() failed: %v", err)
		}
	}

	messages := acc.Messages()
	if len(messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(messages))
	}

	require.Equal(t, expectedMessage, messages[0])

	if acc.FinishReason() != aisdk.FinishReasonStop {
		t.Errorf("Expected finish reason %q, got %q", aisdk.FinishReasonStop, acc.FinishReason())
	}

	require.Equal(t, int64Ptr(10), acc.Usage().PromptTokens)
	require.Equal(t, int64Ptr(90), acc.Usage().CompletionTokens)
}

// Helper function to create a pointer to an int64
func int64Ptr(i int64) *int64 {
	return &i
}

func TestDataStreamAccumulator_ToolCall(t *testing.T) {
	t.Parallel()

	parts := []aisdk.DataStreamPart{
		aisdk.StartStepStreamPart{MessageID: "msg_01PcSiPgKmjGHDU6JNzw5BHP"},
		aisdk.ToolCallStartStreamPart{ToolCallID: "tool_123", ToolName: "get_weather"},
		aisdk.ToolCallDeltaStreamPart{ToolCallID: "tool_123", ArgsTextDelta: "{\"location\":\""},
		aisdk.ToolCallDeltaStreamPart{ToolCallID: "tool_123", ArgsTextDelta: "San Francisco\"}"},
		aisdk.ToolCallStreamPart{
			ToolCallID: "tool_123",
			ToolName:   "get_weather",
			Args:       map[string]any{"location": "San Francisco"},
		},
		aisdk.ToolResultStreamPart{
			ToolCallID: "tool_123",
			Result:     map[string]any{"temperature": 72, "unit": "F"},
		},
		aisdk.FinishStepStreamPart{FinishReason: aisdk.FinishReasonToolCalls, Usage: aisdk.Usage{CompletionTokens: int64Ptr(90)}, IsContinued: false},
		aisdk.FinishMessageStreamPart{FinishReason: aisdk.FinishReasonToolCalls, Usage: aisdk.Usage{CompletionTokens: int64Ptr(90)}},
	}

	expectedMessages := []aisdk.Message{
		{
			ID:   "msg_01PcSiPgKmjGHDU6JNzw5BHP",
			Role: "assistant",
			Parts: []aisdk.Part{
				{
					Type: aisdk.PartTypeStepStart,
				},
				{
					Type: aisdk.PartTypeToolInvocation,
					ToolInvocation: &aisdk.ToolInvocation{
						State:      aisdk.ToolInvocationStateResult,
						ToolCallID: "tool_123",
						ToolName:   "get_weather",
						Args:       map[string]any{"location": "San Francisco"},
						Result:     map[string]any{"temperature": 72, "unit": "F"},
					},
				},
			},
		},
	}

	var acc aisdk.DataStreamAccumulator
	for _, part := range parts {
		err := acc.Push(part)
		require.NoError(t, err, "acc.Push() failed for part type %T", part)
	}

	messages := acc.Messages()
	require.EqualExportedValues(t, expectedMessages, messages)
	require.Equal(t, int64Ptr(90), acc.Usage().CompletionTokens)
}
