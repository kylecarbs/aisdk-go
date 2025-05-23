package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/hamguy/go_grok/pkg/xai"
)

func ToolsToGrok(tools []Tool) ([]map[string]interface{}, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	var grokTools []map[string]interface{}
	for _, tool := range tools {
		if tool.Type != "function" {
			return nil, fmt.Errorf("unsupported tool type: %s", tool.Type)
		}

		grokTool := map[string]interface{}{
			"name":        tool.Function.Name,
			"description": tool.Function.Description,
		}

		if tool.Function.Parameters != nil {
			grokTool["parameters"] = tool.Function.Parameters
		}

		grokTools = append(grokTools, grokTool)
	}

	return grokTools, nil
}

func MessagesToGrok(messages []Message) ([]map[string]interface{}, error) {
	if len(messages) == 0 {
		return nil, nil
	}

	var grokMessages []map[string]interface{}
	for _, message := range messages {
		var content interface{}
		
		if message.Content != "" {
			content = message.Content
		} else if len(message.Parts) > 0 {
			var parts []map[string]interface{}
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					if part.Text != "" {
						content = part.Text
					}
				case PartTypeImage:
					if part.Image != nil && part.Image.Data != "" {
						imgPart := map[string]interface{}{
							"type": "image_url",
							"image_url": map[string]interface{}{
								"url": fmt.Sprintf("data:%s;base64,%s", part.Image.MimeType, part.Image.Data),
							},
						}
						parts = append(parts, imgPart)
					}
				case PartTypeToolResult:
					continue
				}
			}
			
			if len(parts) > 0 {
				content = parts
			}
		}
		
		if content == nil {
			continue
		}
		
		grokMessage := map[string]interface{}{
			"role":    message.Role,
			"content": content,
		}
		
		for _, part := range message.Parts {
			if part.Type == PartTypeToolResult && part.ToolResult != nil {
				grokMessage["tool_calls"] = []map[string]interface{}{
					{
						"id":   part.ToolResult.ToolCallID,
						"type": "function",
						"function": map[string]interface{}{
							"name":      part.ToolResult.ToolName,
							"arguments": part.ToolResult.Args,
						},
					},
				}
			}
		}
		
		grokMessages = append(grokMessages, grokMessage)
	}
	
	return grokMessages, nil
}

func GrokToDataStream(streamChan <-chan *xai.Response) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		var finalReason FinishReason = FinishReasonUnknown
		
		for chunk := range streamChan {
			if chunk == nil {
				continue
			}
			
			if len(chunk.Choices) == 0 {
				continue
			}
			
			choice := chunk.Choices[0]
			
			if choice.FinishReason != "" {
				switch choice.FinishReason {
				case "stop":
					finalReason = FinishReasonStop
				case "length":
					finalReason = FinishReasonLength
				case "tool_calls":
					finalReason = FinishReasonToolCalls
				}
				
				if !yield(FinishStreamPart{Reason: finalReason}, nil) {
					return
				}
				continue
			}
			
			if choice.Delta != nil && choice.Delta.Content != "" {
				if !yield(TextStreamPart{Text: choice.Delta.Content}, nil) {
					return
				}
			}
			
			if choice.Delta != nil && choice.Delta.ToolCalls != nil && len(choice.Delta.ToolCalls) > 0 {
				for _, toolCall := range choice.Delta.ToolCalls {
					if toolCall.Function != nil {
						toolName := toolCall.Function.Name
						arguments := toolCall.Function.Arguments
						
						if toolName != "" || arguments != "" {
							if !yield(ToolCallStreamPart{
								ToolCallID: toolCall.ID,
								ToolName:   toolName,
								Arguments:  arguments,
							}, nil) {
								return
							}
						}
					}
				}
			}
		}
		
		if finalReason == FinishReasonUnknown {
			yield(FinishStreamPart{Reason: finalReason}, nil)
		}
	}
}
