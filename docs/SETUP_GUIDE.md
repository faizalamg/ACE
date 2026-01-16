# ACE Framework Setup Guide

Quick setup and configuration guide for ACE Framework.

*Last updated: 2025-01-15*

## Requirements

- **Python 3.11 or higher**
- API key for your LLM provider (Z.ai GLM, OpenAI, Anthropic, Google, etc.)

Check Python version:
```bash
python --version  # Should show 3.11+
```

---

## Installation

### For Users

```bash
# Basic installation
pip install ace-framework

# With optional features
pip install ace-framework[observability]  # Opik monitoring + cost tracking
pip install ace-framework[browser-use]    # Browser automation
pip install ace-framework[langchain]      # LangChain integration
pip install ace-framework[all]            # All features
```

### For Contributors

```bash
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
uv sync  # Installs everything automatically (10-100x faster than pip)
```

---

## API Key Setup

**ACE uses Z.ai GLM-4.6 by default** (fastest, most cost-effective).

### Option 1: Environment Variable (Recommended)

```bash
# Z.ai GLM (default provider)
export ZAI_API_KEY="your-zai-api-key"

# Or OpenAI
export OPENAI_API_KEY="sk-..."

# Or create .env file
echo "ZAI_API_KEY=your-zai-api-key" > .env
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file
```

### Option 2: Direct in Code

```python
from ace import LiteLLMClient

client = LiteLLMClient(
    model="gpt-4o-mini",
    api_key="your-key-here"  # Not recommended for production
)
```

---

## Provider Examples

### Z.ai GLM (Default)

1. Get API key: [z.ai](https://z.ai)
2. Set key: `export ZAI_API_KEY="your-key"`
3. Use it:
```python
from ace import ACELiteLLM

# Uses Z.ai GLM-4.6 by default
agent = ACELiteLLM()
```

### OpenAI

1. Get API key: [platform.openai.com](https://platform.openai.com)
2. Set key: `export OPENAI_API_KEY="sk-..."`
3. Use it:
```python
from ace import ACELiteLLM
agent = ACELiteLLM(model="gpt-4o-mini")
```

### Anthropic Claude

1. Get API key: [console.anthropic.com](https://console.anthropic.com)
2. Set key: `export ANTHROPIC_API_KEY="sk-ant-..."`
3. Use it:
```python
client = LiteLLMClient(model="claude-3-5-sonnet-20241022")
```

### Google Gemini

1. Get API key: [makersuite.google.com](https://makersuite.google.com)
2. Set key: `export GOOGLE_API_KEY="AIza..."`
3. Use it:
```python
client = LiteLLMClient(model="gemini-pro")
```

### Local Models (Ollama)

1. Install Ollama: [ollama.ai](https://ollama.ai)
2. Pull model: `ollama pull llama2`
3. Use it:
```python
agent = ACELiteLLM(model="ollama/llama2")
```

### Local Models (LM Studio)

1. Install LM Studio: [lmstudio.ai](https://lmstudio.ai)
2. Load model and start server (default port 1234)
3. Use it:
```python
agent = ACELiteLLM(
    model="openai/local-model",
    api_base="http://localhost:1234/v1"
)
```

See [examples/LMstudio/](../examples/LMstudio/) for complete setup guide.

**Supported Providers:** 100+ via LiteLLM (AWS Bedrock, Azure, Cohere, Hugging Face, etc.)

---

## Advanced Configuration

### Custom LLM Parameters

```python
from ace import LiteLLMClient

client = LiteLLMClient(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=2048,
    timeout=60  # seconds
)
```

### Production Monitoring (Opik)

```bash
pip install ace-framework[observability]
```

Opik automatically tracks:
- Token usage per LLM call
- Cost per operation
- Generator/Reflector/Curator performance
- Playbook evolution over time

View dashboard: [comet.com/opik](https://www.comet.com/opik)

### Playbook Storage

```python
from ace import Playbook

# Save playbook
playbook.save_to_file("my_playbook.json")

# Load playbook
playbook = Playbook.load_from_file("my_playbook.json")

# For production: Use database storage
# PostgreSQL, SQLite, or vector stores supported
```

### Retry Configuration

```python
from ace import Generator

# Configure JSON parsing retries
generator = Generator(
    llm,
    retry_prompt="\n\nPlease return valid JSON only."
)
```

### Checkpoint Saving

```python
from ace import OfflineAdapter

adapter = OfflineAdapter(playbook, generator, reflector, curator)

# Save playbook every 10 samples during training
results = adapter.run(
    samples,
    environment,
    checkpoint_interval=10,
    checkpoint_dir="./checkpoints"
)
```

---

## Advanced Configuration

### Embedding & Vector Database Configuration

ACE uses **centralized configuration** for embedding and Qdrant settings via `ace/config.py`.

#### Environment Variables

```bash
# Embedding Server Configuration
export ACE_EMBEDDING_URL="http://localhost:1234"
export ACE_EMBEDDING_MODEL="qwen/qwen3-embedding-8b"
export ACE_EMBEDDING_DIM="4096"

# Qdrant Configuration
export ACE_QDRANT_URL="http://localhost:6333"
export ACE_UNIFIED_COLLECTION="ace_memories_hybrid"
```

#### Configuration Defaults

If environment variables are not set, ACE uses these defaults:

| Variable | Default Value | Purpose |
|----------|---------------|---------|
| `ACE_EMBEDDING_URL` | `http://localhost:1234` | LM Studio or embedding server URL |
| `ACE_EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` | Embedding model name (4096 dims) |
| `ACE_EMBEDDING_DIM` | `4096` | Embedding vector dimension |
| `ACE_QDRANT_URL` | `http://localhost:6333` | Qdrant vector database URL |
| `ACE_UNIFIED_COLLECTION` | `ace_memories_hybrid` | Qdrant collection name |

#### Programmatic Configuration

```python
from ace.config import EmbeddingConfig, QdrantConfig
from ace.unified_memory import UnifiedMemoryIndex

# Custom embedding configuration
embedding_config = EmbeddingConfig(
    url="http://custom-server:1234",
    model="custom-embedding-model",
    dimension=768
)

# Custom Qdrant configuration
qdrant_config = QdrantConfig(
    url="http://qdrant-server:6333",
    collection_name="custom_collection"
)

# Use with UnifiedMemoryIndex
index = UnifiedMemoryIndex(
    embedding_config=embedding_config,
    qdrant_config=qdrant_config
)
```

#### Local Embedding Server Setup (LM Studio)

**Recommended Model**: `Qwen3-Embedding-8B-Q5_K_M.gguf` (4096 dimensions)

1. Download [LM Studio](https://lmstudio.ai/)
2. Search and download: `Qwen3-Embedding-8B-Q5_K_M`
3. Load model in LM Studio Server tab
4. Start server on port 1234
5. Set environment variable:
   ```bash
   export ACE_EMBEDDING_URL="http://localhost:1234"
   ```

**Note**: Qwen embedding models require `</s>` EOS token for optimal quality. ACE automatically appends this token in all embedding functions.

#### Qdrant Setup

**Option 1: Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option 2: Local Binary**
Download from [qdrant.tech/documentation/quick-start](https://qdrant.tech/documentation/quick-start/)

**Verify Connection:**
```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
print(client.get_collections())  # Should list collections
```

---

## Troubleshooting

### Import Errors

```bash
# Upgrade to latest version
pip install --upgrade ace-framework

# Check installation
pip show ace-framework
```

### API Key Not Working

```bash
# Verify key is set
echo $OPENAI_API_KEY

# Test different model
from ace import LiteLLMClient
client = LiteLLMClient(model="gpt-3.5-turbo")  # Cheaper for testing
```

### Rate Limits

```python
from ace import LiteLLMClient

# Add delays between calls
import time
time.sleep(1)  # 1 second between calls

# Or use a cheaper/faster model
client = LiteLLMClient(model="gpt-3.5-turbo")
```

### JSON Parse Failures

```python
# Increase max_tokens for Curator/Reflector
from ace import Curator, Reflector

llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)  # Higher limit
curator = Curator(llm)
reflector = Reflector(llm)
```

---

## Need More Help?

- **GitHub Issues:** [github.com/kayba-ai/agentic-context-engine/issues](https://github.com/kayba-ai/agentic-context-engine/issues)
- **Documentation:** [Complete Guide](COMPLETE_GUIDE_TO_ACE.md), [Quick Start](QUICK_START.md), [Integration Guide](INTEGRATION_GUIDE.md)

---

**Next Steps:** Check out the [Quick Start Guide](QUICK_START.md) to build your first self-learning agent!

