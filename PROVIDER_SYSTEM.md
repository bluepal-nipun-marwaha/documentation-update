# Multi-Provider LLM and Embeddings System

This document describes the new multi-provider system that allows you to select between different LLM and embeddings providers.

## Overview

The system now supports multiple providers for both LLM and embeddings:

### LLM Providers
- **OpenAI** (GPT-4, GPT-3.5-turbo)
- **Anthropic** (Claude-3-sonnet, Claude-3-haiku)
- **Mistral** (Mistral Large 2)
- **Gemma** (via Google AI Studio)
- **Cohere** (Command R+)
- **Ollama** (Local models with GPU acceleration)

### Embeddings Providers
- **OpenAI** (text-embedding-3-large, text-embedding-3-small)
- **Cohere** (embed-english-v3.0)
- **Gemini** (text-embedding-004)
- **Nomic** (nomic-embed-text) - via Ollama
- **E5** (e5-large-v2) - via Ollama
- **Sentence Transformers** (all-MiniLM-L6-v2) - via Ollama

## Configuration

### Basic Configuration

Set your provider preferences in the `.env` file:

```bash
# LLM Provider Selection
LLM_PROVIDER=openai  # or anthropic, mistral, gemma, cohere, ollama
LLM_MODEL=gpt-4

# Embeddings Provider Selection
EMBEDDINGS_PROVIDER=openai  # or cohere, gemini, nomic, e5, sentence_transformers
EMBEDDINGS_MODEL=text-embedding-3-large
```

### Provider-Specific Configuration

#### OpenAI
```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-large
```

#### Anthropic
```bash
ANTHROPIC_API_KEY=your-api-key-here
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

#### Mistral
```bash
MISTRAL_API_KEY=your-api-key-here
MISTRAL_MODEL=mistral-large-latest
```

#### Gemma
```bash
GEMMA_API_KEY=your-api-key-here
GEMMA_MODEL=gemma-7b-it
```

#### Cohere
```bash
COHERE_API_KEY=your-api-key-here
COHERE_MODEL=command-r-plus
COHERE_EMBEDDINGS_MODEL=embed-english-v3.0
```

#### Ollama (Local)
```bash
OLLAMA_BASE_URL=http://localhost:11434
USE_GPU=true
GPU_LAYERS=-1
```

## Model Management

### GitHub Model Distribution

The system supports downloading models from GitHub releases for faster setup:

```bash
USE_GITHUB_MODELS=true
GITHUB_MODEL_REPO=your-org/rework-models
LOCAL_MODEL_CACHE=~/.cache/rework-models
```

### Model Versions

Models are pinned to specific versions for reproducibility:

- `qwen2.5-7b`: v1.2.3
- `gemma-7b`: v2.1.0
- `mistral-7b`: v1.0.5
- `nomic-embed-text`: v1.0.0
- `e5-large-v2`: v1.0.0
- `all-MiniLM-L6-v2`: v1.0.0

## Usage Examples

### Using OpenAI
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-large
OPENAI_API_KEY=your-key-here
```

### Using Anthropic
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-large
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

### Using Local Ollama
```bash
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
EMBEDDINGS_PROVIDER=nomic
EMBEDDINGS_MODEL=nomic-embed-text
USE_GPU=true
```

### Hybrid Setup (Remote LLM + Local Embeddings)
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
EMBEDDINGS_PROVIDER=nomic
EMBEDDINGS_MODEL=nomic-embed-text
OPENAI_API_KEY=your-key-here
```

## Error Handling

The system provides clear error messages when providers fail:

- **Provider not available**: Missing API keys or service unavailable
- **Model not found**: Invalid model name or version
- **API errors**: Network issues or rate limits
- **Configuration errors**: Invalid provider settings

## Performance Considerations

### Local vs Remote

- **Local (Ollama)**: Privacy, no API costs, GPU acceleration
- **Remote**: Better models, no local resources, API costs

### Embeddings Dimensions

Different providers have different embedding dimensions:

- OpenAI: 1536-3072 dimensions
- Cohere: 1024 dimensions
- Gemini: 768 dimensions
- Nomic: 768 dimensions
- E5: 1024 dimensions
- Sentence Transformers: 384 dimensions

## Migration from Old System

The new system is backward compatible. Existing Ollama configurations will continue to work:

```bash
# Old configuration (still works)
OLLAMA_BASE_URL=http://localhost:11434
USE_GPU=true

# New configuration (recommended)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
EMBEDDINGS_PROVIDER=nomic
EMBEDDINGS_MODEL=nomic-embed-text
```

## Troubleshooting

### Common Issues

1. **Provider not available**: Check API keys and service status
2. **Model download fails**: Check GitHub repository access
3. **GPU not working**: Verify Ollama GPU configuration
4. **Embeddings mismatch**: Ensure consistent provider for storage and retrieval

### Debug Mode

Enable debug logging to see provider initialization:

```bash
LOGGING_LEVEL=DEBUG
```

## Future Enhancements

- Custom model fine-tuning support
- Provider health monitoring
- Automatic failover between providers
- Cost tracking and optimization
- Model performance benchmarking
