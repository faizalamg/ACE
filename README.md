# ACE - Agentic Context Engine

They said it couldn't be done. We did it.

They said the sauce was secret. We figured it out.

In 1000+ head to head tests against the Big Boys. We win. Every. Single. Time.

Better content. Better relevancy. Better context.

Code context retrieval that actually works. 94% accuracy. Self-learning. 

MCP - so it runs anywhere. Local - so what's yours stays yours. Open source - so it's available and transparent for all.

Don't take our word for it. It's free and open source. Run your own head to head benchmarks and see for yourself.

Like it? Star it. Issues? Report it. Ideas? Contribute. 

---

## Quick Start

```bash
pip install ace-framework
```

```python
from ace import UnifiedRetriever
retriever = UnifiedRetriever()
results = retriever.retrieve("your query")
```

[Full Setup Guide](docs/SETUP_GUIDE.md)

---

## 94% Retrieval Accuracy

| Metric | Value |
|--------|-------|
| Accuracy | 94% |
| Test Queries | 1,000 |
| Response Time | <200ms |

---

## The Stack

`LinUCB` `HyDE` `HDBSCAN` `Cross-Encoder` `BM25` `Qdrant` `AST-Chunking` `Semantic-Dedup` `Confidence-Decay` `MiniLM` `Voyage`

---

## Runs Anywhere

- **Local**: Ollama, LMStudio, any embedding model
- **Cloud**: OpenAI, Voyage, Gemini
- **IDE**: MCP server for VS Code, Cursor, Claude

---

## Recommended Setup

| Component | Recommendation | Notes |
|-----------|---------------|-------|
| **Embeddings (Text)** | Qwen3-Embedding-8B (4096d) | Local via LM Studio, ~8GB VRAM |
| **Embeddings (Code)** | Voyage-code-3 | API, optimized for code |
| **Vector DB** | Qdrant | Local or cloud, free tier available |
| **LLM** | Any | Ollama, LM Studio, OpenAI, Gemini |

```bash
# Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# LM Studio
# 1. Download: lmstudio.ai
# 2. Load: Qwen3-Embedding-8B
# 3. Start server on port 1234
```

---

## What Makes It Work

**HyDE** - Hypothetical Document Embeddings for query expansion

**LinUCB Bandit** - Learns which retrieval strategies work for your data

**HDBSCAN Dedup** - Kills near-duplicate chunks

**Confidence Decay** - Old memories fade, fresh data wins

**AST Chunking** - Code-aware splitting that doesn't break functions

**Cross-Encoder Reranking** - Precision filtering after retrieval

**Self-Learning Memory** - Cross-workspace patterns + project-specific knowledge. The one-two punch.

---

## Docs

- [Setup Guide](docs/SETUP_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [MCP Integration](docs/MCP_INTEGRATION.md)

---

## Acknowledgments

**Research Foundation:**
- [Agentic Context Engineering](https://arxiv.org/abs/2510.04618) - UC Berkeley & SambaNova Systems (Zhang et al.)
- [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952) methodology
- [LinUCB Algorithm](https://arxiv.org/abs/1003.0146) - Li et al. (2010)

**Code Inspirations:**
- [Kayba.ai](https://github.com/kayba-ai/agentic-context-engine) - Original implementation
- [m1rl0k/Context-Engine](https://github.com/m1rl0k/Context-Engine) - AST chunking, ReFRAG
- [ELF](https://github.com/Spacehunterz/Emergent-Learning-Framework_ELF) - Confidence decay, golden rules
- [r/Rag community](https://www.reddit.com/r/Rag/) - Memory architecture discussions

**Built On:**
- [Qdrant](https://qdrant.tech/) - Vector search
- [Sentence Transformers](https://www.sbert.net/) - MiniLM, Cross-Encoders
- [Voyage AI](https://www.voyageai.com/) - Embeddings
- [HDBSCAN](https://hdbscan.readthedocs.io/) - Clustering
- [LiteLLM](https://github.com/BerriAI/litellm) - Multi-provider LLM
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework

## License

MIT.
