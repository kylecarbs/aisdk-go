package aisdk

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/hamguy/go_grok/pkg/xai"
)

func ToolsToGrok(tools []Tool) ([]map[string]interface{}, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	var grokTools []map[string]interface{}
	for _, tool := range tools {
		grokTool := map[string]interface{}{
			"name":        tool.Name,
			"description": tool.Description,
		}

		if tool.Schema.Properties != nil {
			schemaParams := map[string]interface{}{
				"type":       "object",
				"properties": tool.Schema.Properties,
			}
			if len(tool.Schema.Required) > 0 {
				schemaParams["required"] = tool.Schema.Required
			}
			grokTool["parameters"] = schemaParams
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
				case PartTypeFile:
					if part.Data != nil && len(part.Data) > 0 {
						imgPart := map[string]interface{}{
							"type": "image_url",
							"image_url": map[string]interface{}{
								"url": fmt.Sprintf("data:%s;base64,%s", part.MimeType, base64.StdEncoding.EncodeToString(part.Data)),
							},
						}
						parts = append(parts, imgPart)
					}
				case PartTypeToolInvocation:
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
			if part.Type == PartTypeToolInvocation && part.ToolInvocation != nil && part.ToolInvocation.State == ToolInvocationStateResult {
				grokMessage["tool_calls"] = []map[string]interface{}{
					{
						"id":   part.ToolInvocation.ToolCallID,
						"type": "function",
						"function": map[string]interface{}{
							"name":      part.ToolInvocation.ToolName,
							"arguments": part.ToolInvocation.Args,
						},
					},
				}
			}
		}
		
		grokMessages = append(grokMessages, grokMessage)
	}
	
	return grokMessages, nil
}

func GrokToDataStream(streamChan <-chan *xai.ChatCompletionResponse) DataStream {
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
				
				if !yield(FinishStepStreamPart{
					IsContinued:  false,
					FinishReason: finalReason,
				}, nil) {
					return
				}
				continue
			}
			
			if choice.Delta != nil && choice.Delta.Content != "" {
				if !yield(TextStreamPart{Content: choice.Delta.Content}, nil) {
					return
				}
			}
			
			if choice.Delta != nil && choice.Delta.ToolCalls != nil && len(choice.Delta.ToolCalls) > 0 {
				for _, toolCall := range choice.Delta.ToolCalls {
					toolName := toolCall.Function.Name
					var args map[string]any
					if toolCall.Function.Arguments != nil {
						if argsStr, ok := toolCall.Function.Arguments.(string); ok {
							if err := json.Unmarshal([]byte(argsStr), &args); err != nil {
								args = map[string]any{"text": argsStr}
							}
						}
					}
					
					if toolName != "" || len(args) > 0 {
						if !yield(ToolCallStreamPart{
							ToolCallID: toolCall.ID,
							ToolName:   toolName,
							Args:       args,
						}, nil) {
							return
						}
					}
				}
			}
		}
		
		yield(FinishMessageStreamPart{
			FinishReason: finalReason,
		}, nil)
	}
}
