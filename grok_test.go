package aisdk_test

import (
	"os"
	"testing"

	"github.com/hamguy/go_grok/pkg/xai"
	"github.com/kylecarbs/aisdk-go"
	"github.com/stretchr/testify/require"
)

func TestGrokToDataStream(t *testing.T) {
	t.Parallel()

	mockResponse := &xai.ChatCompletionResponse{
		Choices: []xai.Choice{
			{
				Delta: &xai.Message{
					Content: "Hello, world!",
				},
			},
		},
	}

	streamChan := make(chan *xai.ChatCompletionResponse, 1)
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
			Name:        "get_weather",
			Description: "Get the weather for a location",
			Schema: aisdk.Schema{
				Required: []string{"location"},
				Properties: map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "The location to get weather for",
					},
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

	apiKey := os.Getenv("GROK_API_KEY")
	if apiKey == "" {
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

	client := xai.NewClient(apiKey, "grok-2-1212")
	
	streamTrue := true
	req := &xai.ChatCompletionRequest{
		Model:    "grok-2-1212",
		Messages: grokMessages,
		Stream:   &streamTrue,
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
