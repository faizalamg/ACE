# Technology Stack

**Analysis Date:** 2026-01-19

## Languages

**Primary:**
- Python 3.11+ - Core framework implementation

**Secondary:**
- Not applicable

## Runtime

**Environment:**
- Python 3.11, 3.12 supported

**Package Manager:**
- uv - Primary package manager (uv.lock present)
- setuptools - Build backend (>=61.0)
- Lockfile: uv.lock present

## Frameworks

**Core:**
- FastMCP - Model Context Protocol server framework for VS Code/Claude Desktop integration
- LiteLLM 1.78.0+ - Multi-provider LLM client abstraction

**LLM/Framework Integration:**
- langchain-core 0.3.79+ - LangChain abstraction layer
- langchain-openai 0.3.35+ - OpenAI LangChain integration
- langchain-litellm 0.2.0+ - LiteLLM LangChain integration
- browser-use 0.9.1+ - Browser automation framework

**ML/ML:**
- sentence-transformers 5.2.0+ - Sentence embeddings and cross-encoder models
- transformers 4.30.0+ (optional) - Hugging Face models
- torch 2.0.0+ (optional) - PyTorch for transformer models
- accelerate 0.20.0+ (optional) - Model acceleration
- optimum 2.0.0+ with ONNX Runtime - ONNX model optimization
- onnxruntime-directml 1.23.0+ - DirectML backend for ONNX

**Build/Dev:**
- black 23.0.0+ - Code formatting
- mypy 1.0.0+ - Static type checking
- pre-commit 3.0.0+ - Git hooks
- pytest 7.0.0+ - Testing framework
- pytest-asyncio 0.21.0+ - Async test support
- pytest-cov 4.0.0+ - Coverage reporting

## Key Dependencies

**Critical:**
- pydantic 2.0.0+ - Data validation and settings management
- python-dotenv 1.0.0+ - Environment configuration
- pyjwt 2.10.1+ - JWT token handling
- python-toon 0.1.0+ - Tool output normalization

**Vector Database:**
- qdrant-client 1.16.1+ - Vector storage and similarity search

**Embedding/API:**
- voyageai 0.3.0+ - Code embedding API (voyage-code-3)

**Utilities:**
- tenacity 8.0.0+ - Retry logic and resilience

**Observability (Optional):**
- httpx 0.27.0-0.29.0 - Async HTTP client for Qdrant REST API
- opik 1.8.0+ - LLM tracing and monitoring
- prometheus-client 0.23.1+ - Metrics collection

**Code Analysis (Optional):**
- tree-sitter 0.23.0+ - AST-based code parsing
- tree-sitter-python 0.23.0+ - Python AST
- tree-sitter-typescript 0.23.0+ - TypeScript AST
- tree-sitter-go 0.23.0+ - Go AST
- tree-sitter-javascript 0.23.0+ - JavaScript AST

**Demo/Benchmark (Optional):**
- rich 13.0.0+ - Terminal output formatting
- datasets 2.0.0+ - Benchmark datasets
- pandas 2.0.0+ - Data manipulation
- openpyxl 3.0.0+ - Excel file handling
- pyyaml 6.0.0+ - YAML configuration
- playwright 1.40.0+ - Browser automation (alternative to browser-use)

## Configuration

**Environment:**
- python-dotenv for .env file loading
- Centralized config via `ace/config.py` dataclasses

**Build:**
- setuptools.build_meta backend
- pyproject.toml with full dependency specifications

## Platform Requirements

**Development:**
- Python 3.11+ (tested on 3.11, 3.12)
- Optional: Local GPU for embedding models (~8GB VRAM for Qwen3-Embedding-8B)

**Production:**
- Docker: Qdrant can run in container
- Qdrant: Local or cloud (http://localhost:6333 default)
- MCP-compatible: VS Code, Claude Desktop, Cursor

---

*Stack analysis: 2026-01-19*
