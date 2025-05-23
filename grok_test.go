package aisdk_test

import (
	"testing"

	"github.com/hamguy/go_grok/pkg/xai"
	"github.com/kylecarbs/aisdk-go"
	"github.com/stretchr/testify/require"
)

func TestGrokToDataStream(t *testing.T) {
	t.Parallel()

	mockResponse := &xai.Response{
		Choices: []*xai.Choice{
			{
				Delta: &xai.Delta{
					Content: "Hello, world!",
				},
			},
		},
	}

	streamChan := make(chan *xai.Response, 1)
	streamChan <- mockResponse
	close(streamChan)

	var acc aisdk.DataStreamAccumulator
	stream := aisdk.GrokToDataStream(streamChan)
	stream = stream.WithAccumulator(&acc)

	for _, err := range stream {
		require.NoError(t, err)
	}

	messages := acc.Messages()
	require.Len(t, messages, 1)
	require.Equal(t, "assistant", messages[0].Role)
	require.Equal(t, "Hello, world!", messages[0].Content)
}

func TestMessagesToGrok(t *testing.T) {
	t.Parallel()

	messages := []aisdk.Message{
		{
			Role:    "system",
			Content: "You are a helpful assistant.",
		},
		{
			Role:    "user",
			Content: "Hello, how are you?",
		},
	}

	grokMessages, err := aisdk.MessagesToGrok(messages)
	require.NoError(t, err)
	require.Len(t, grokMessages, 2)
	require.Equal(t, "system", grokMessages[0]["role"])
	require.Equal(t, "You are a helpful assistant.", grokMessages[0]["content"])
	require.Equal(t, "user", grokMessages[1]["role"])
	require.Equal(t, "Hello, how are you?", grokMessages[1]["content"])
}

func TestToolsToGrok(t *testing.T) {
	t.Parallel()

	tools := []aisdk.Tool{
		{
			Type: "function",
			Function: aisdk.Function{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The location to get weather for",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	grokTools, err := aisdk.ToolsToGrok(tools)
	require.NoError(t, err)
	require.Len(t, grokTools, 1)
	require.Equal(t, "get_weather", grokTools[0]["name"])
	require.Equal(t, "Get the weather for a location", grokTools[0]["description"])
	require.NotNil(t, grokTools[0]["parameters"])
}

func TestMessagesToGrok_Live(t *testing.T) {
	t.Parallel()

	if aisdk.GetEnv("GROK_API_KEY") == "" {
		t.Skip("GROK_API_KEY is not set")
	}

	messages := []aisdk.Message{
		{
			Role:    "system",
			Content: "You are a helpful assistant.",
		},
		{
			Role:    "user",
			Content: "Hello, how are you?",
		},
	}

	grokMessages, err := aisdk.MessagesToGrok(messages)
	require.NoError(t, err)
	require.Len(t, grokMessages, 2)

	client := xai.NewClient(aisdk.GetEnv("GROK_API_KEY"))
	req := &xai.ChatCompletionRequest{
		Model:    xai.Grok212,
		Messages: grokMessages,
		Stream:   true,
	}

	streamChan, err := client.CreateChatCompletionStream(req)
	require.NoError(t, err)

	var acc aisdk.DataStreamAccumulator
	stream := aisdk.GrokToDataStream(streamChan.Stream)
	stream = stream.WithAccumulator(&acc)

	for _, err := range stream {
		require.NoError(t, err)
	}

	accMessages := acc.Messages()
	require.Len(t, accMessages, 1)
	require.Equal(t, "assistant", accMessages[0].Role)
	require.NotEmpty(t, accMessages[0].Content)
}
