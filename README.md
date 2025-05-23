# aisdk-go

[![GitHub Release](https://img.shields.io/github/v/release/kylecarbs/aisdk-go?color=6b9ded&sort=semver)](https://github.com/kylecarbs/aisdk-go/releases)
[![GoDoc](https://godoc.org/github.com/kylecarbs/aisdk-go?status.svg)](https://godoc.org/github.com/kylecarbs/aisdk-go)
[![CI Status](https://github.com/kylecarbs/aisdk-go/workflows/ci/badge.svg)](https://github.com/kylecarbs/aisdk-go/actions)

> [!WARNING]  
> This library is super new and may change a lot.

A Go implementation of Vercel's AI SDK [Data Stream Protocol](https://sdk.vercel.ai/docs/ai-sdk-ui/stream-protocol#data-stream-example).

- Supports OpenAI, Google, Anthropic (with Bedrock support), and Grok
- Examples for integrating `useChat`
- Chain tool usage in Go, just like `maxSteps`

```ts
// frontend.tsx

const { messages } = useChat({
    // Points to our Go backend!
    api: "/api/chat",
})
```

```go
// backend.go

// Accept the POST request...
var req *aisdk.Chat

messages, err := aisdk.MessagesToOpenAI(req.Messages)
if err != nil {
    http.Error(w, err.Error(), http.StatusInternalServerError)
    return
}

// Convert the http.ResponseWriter to a Data Stream.
dataStream := aisdk.NewDataStream(w)
stream := openaiClient.Chat.Completions.NewStreaming(...)

aisdk.PipeOpenAIToDataStream(stream, dataStream)
```

## Development

Run tests with `go test`. Start the `useChat` demo with:

```bash
# any or all of these can be set
export OPENAI_API_KEY=<api-key>
export ANTHROPIC_API_KEY=<api-key>
export GOOGLE_API_KEY=<api-key>
export GROK_API_KEY=<api-key>

cd demo
bun i
bun dev
```
