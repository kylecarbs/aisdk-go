package aisdk_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/coder/aisdk-go"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestGoogleToDataStream(t *testing.T) {
	t.Parallel()

	// googleResponses are hardcoded responses from the Google AI Stream endpoint.
	googleResponses := `data: {"candidates": [{"content": {"parts": [{"text": "Here"}],"role": "model"}}],"usageMetadata": {"promptTokenCount": 10,"totalTokenCount": 10,"promptTokensDetails": [{"modality": "TEXT","tokenCount": 10}]},"modelVersion": "gemini-2.0-flash","responseId": "ibRCaOfLGfuQ1PIPqNma8Aw"}

data: {"candidates": [{"content": {"parts": [{"text": " you go:\n\nEnglish: potato\nSpanish: patata\nFrench: pom"}],"role": "model"}}],"usageMetadata": {"promptTokenCount": 10,"totalTokenCount": 10,"promptTokensDetails": [{"modality": "TEXT","tokenCount": 10}]},"modelVersion": "gemini-2.0-flash","responseId": "ibRCaOfLGfuQ1PIPqNma8Aw"}

data: {"candidates": [{"content": {"parts": [{"text": "me de terre\nGerman: Kartoffel\nItalian: patata\nJapanese: ジャ"}],"role": "model"}}],"usageMetadata": {"promptTokenCount": 10,"totalTokenCount": 10,"promptTokensDetails": [{"modality": "TEXT","tokenCount": 10}]},"modelVersion": "gemini-2.0-flash","responseId": "ibRCaOfLGfuQ1PIPqNma8Aw"}

data: {"candidates": [{"content": {"parts": [{"text": "ガイモ (jagaimo)\nRussian: картофель (kartofel')\n"}],"role": "model"},"finishReason": "STOP"}],"usageMetadata": {"promptTokenCount": 10,"candidatesTokenCount": 49,"totalTokenCount": 59,"promptTokensDetails": [{"modality": "TEXT","tokenCount": 10}],"candidatesTokensDetails": [{"modality": "TEXT","tokenCount": 49}]},"modelVersion": "gemini-2.0-flash","responseId": "ibRCaOfLGfuQ1PIPqNma8Aw"}`

	// Parse the SSE format manually since Google doesn't expose its SSE parsing :(
	// See https://github.com/googleapis/go-genai/blob/v1.10.0/api_client.go#L44
	lines := strings.Split(googleResponses, "\n")
	var responses []*genai.GenerateContentResponse

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data: ") {
			jsonData := strings.TrimPrefix(line, "data: ")
			if len(jsonData) == 0 {
				continue // Skip empty data lines
			}
			var resp genai.GenerateContentResponse
			err := json.Unmarshal([]byte(jsonData), &resp)
			require.NoError(t, err)
			responses = append(responses, &resp)
		}
	}

	// Create an iterator from the parsed responses
	mockStream := func(yield func(*genai.GenerateContentResponse, error) bool) {
		for _, resp := range responses {
			if !yield(resp, nil) {
				return
			}
		}
	}

	var acc aisdk.DataStreamAccumulator
	stream := aisdk.GoogleToDataStream(mockStream)
	stream = stream.WithToolCalling(func(toolCall aisdk.ToolCall) any {
		return map[string]any{"message": "Message printed to the console"}
	})
	stream = stream.WithAccumulator(&acc)
	for _, err := range stream {
		require.NoError(t, err)
	}

	// Verify that we got messages and they have the expected content
	messages := acc.Messages()
	require.Len(t, messages, 1)

	expectedContent := "Here you go:\n\nEnglish: potato\nSpanish: patata\nFrench: pomme de terre\nGerman: Kartoffel\nItalian: patata\nJapanese: ジャガイモ (jagaimo)\nRussian: картофель (kartofel')\n"

	msg := messages[0]
	require.Equal(t, "assistant", msg.Role)
	require.Equal(t, expectedContent, msg.Content)
	require.Len(t, msg.Parts, 2) // step-start, text (accumulated)

	// Check step start part
	require.Equal(t, aisdk.PartTypeStepStart, msg.Parts[0].Type)

	// Check text part (accumulated across all chunks)
	require.Equal(t, aisdk.PartTypeText, msg.Parts[1].Type)
	require.Equal(t, expectedContent, msg.Parts[1].Text)

	// Test conversion back to Google format
	googleContents, err := aisdk.MessagesToGoogle(messages)
	require.NoError(t, err)

	// We expect one content block with just text
	require.Len(t, googleContents, 1)

	// Check the content (assistant message with text only)
	content := googleContents[0]
	require.Equal(t, "model", content.Role)
	require.Len(t, content.Parts, 1) // just text part

	// Check text part (accumulated text)
	require.Equal(t, expectedContent, content.Parts[0].Text)
}

func TestMessagesToGoogle_Live(t *testing.T) {
	t.Parallel()
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		t.Skip("GOOGLE_API_KEY is not set")
	}

	// Create a Google AI client
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	require.NoError(t, err)

	prompt := "use the 'print' tool to print 'Hello, world!' and then show the result"
	// Test messages with a simple request
	messages := []aisdk.Message{
		{
			Role: "user",
			Parts: []aisdk.Part{
				{Text: prompt, Type: aisdk.PartTypeText},
			},
		},
	}

	// Convert messages to Google format
	contents, err := aisdk.MessagesToGoogle(messages)
	require.NoError(t, err)
	require.Len(t, contents, 1)
	require.Len(t, contents[0].Parts, 1)
	require.Equal(t, contents[0].Parts[0].Text, prompt)

	_, err = genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  os.Getenv("GOOGLE_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})
	require.NoError(t, err)

	stream := client.Models.GenerateContentStream(
		ctx,
		"gemini-2.0-flash",
		contents,
		nil,
	)

	dataStream := aisdk.GoogleToDataStream(stream)
	var streamErr error
	dataStream(func(part aisdk.DataStreamPart, err error) bool {
		if err != nil {
			streamErr = err
			return false
		}
		return true
	})
	require.NoError(t, streamErr)
}
