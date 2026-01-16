# ACE Framework Examples

Navigation guide for all ACE examples. Each directory has its own detailed README.

*Last updated: 2025-01-15*

## Getting Started

**New to ACE?** Start with these:

- **[simple_ace_example.py](simple_ace_example.py)** - Minimal ACE usage (5 minutes)
- **[seahorse_emoji_ace.py](seahorse_emoji_ace.py)** - Self-reflection demo
- **[Quick Start Guide](../docs/QUICK_START.md)** - Step-by-step tutorial

## Quick Start Templates

Copy-paste starting points for different providers:

- **[starter-templates/quickstart_litellm.py](starter-templates/quickstart_litellm.py)** - LiteLLM (100+ providers)
- **[starter-templates/langchain_starter_template.py](starter-templates/langchain_starter_template.py)** - LangChain integration
- **[starter-templates/ollama_starter_template.py](starter-templates/ollama_starter_template.py)** - Local Ollama models
- **[LMstudio/](LMstudio/)** - Local LM Studio models (new!)
- **[zai_glm_example.py](zai_glm_example.py)** - Z.ai GLM integration (default provider)

## 🧩 Integrations

Add ACE learning to existing systems:

### Browser Automation (browser-use)
**[browser-use/](browser-use/)** - Self-improving browser agents

- [simple_ace_agent.py](browser-use/simple_ace_agent.py) - Basic ACEAgent usage
- [domain-checker/](browser-use/domain-checker/) - Domain availability automation
- [form-filler/](browser-use/form-filler/) - Form filling automation
- [online-shopping/](browser-use/online-shopping/) - E-commerce automation

📖 See [browser-use/README.md](browser-use/README.md) for full guide

### LangChain Integration
**[langchain/](langchain/)** - Wrap LangChain chains/agents with learning

- [simple_chain_example.py](langchain/simple_chain_example.py) - Basic chain + ACE
- [agent_with_tools_example.py](langchain/agent_with_tools_example.py) - Agent with tools

📖 See [langchain/README.md](langchain/README.md) for patterns

### Custom Integration
**[custom_integration_example.py](custom_integration_example.py)** - Pattern for any agent

Shows the three-step integration: Inject → Execute → Learn

## 📊 Advanced Topics

### Production Learning
**[helicone/](helicone/)** - Learn from Helicone observability logs

- Parse production LLM traces
- Replay-based learning (cost-effective)
- Tool selection analysis

📖 See [helicone/README.md](helicone/README.md)

### Prompt Engineering
**[prompts/](prompts/)** - Compare ACE prompt versions

- [compare_v1_v2_prompts.py](prompts/compare_v1_v2_prompts.py) - v1.0 vs v2.0
- [compare_v2_v2_1_prompts.py](prompts/compare_v2_v2_1_prompts.py) - v2.0 vs v2.1 (recommended)
- [advanced_prompts_v2.py](prompts/advanced_prompts_v2.py) - Advanced techniques

### Playbook Management
- **[playbook_persistence.py](playbook_persistence.py)** - Save and load learned strategies

## Examples by Use Case

| Use Case | Example |
|----------|---------|
| Q&A systems | [simple_ace_example.py](simple_ace_example.py) |
| Browser automation | [browser-use/](browser-use/) |
| LangChain workflows | [langchain/](langchain/) |
| Custom agents | [custom_integration_example.py](custom_integration_example.py) |
| Production learning | [helicone/](helicone/) |
| Prompt optimization | [prompts/](prompts/) |
| Local models (LM Studio) | [LMstudio/](LMstudio/) |
| Z.ai GLM integration | [zai_glm_example.py](zai_glm_example.py) |

## Quick Start

```bash
# 1. Install
pip install ace-framework

# 2. Set API key (Z.ai GLM is default)
export ZAI_API_KEY="your-zai-api-key"
# Or use OpenAI: export OPENAI_API_KEY="your-api-key"

# 3. Run example
python examples/simple_ace_example.py

# Local model example (LM Studio)
python examples/LMstudio/lmstudio_starter_template.py

# Browser examples (contributors: uv sync --group demos)
uv run python examples/browser-use/simple_ace_agent.py
```

## 📚 Documentation

- **[Quick Start Guide](../docs/QUICK_START.md)** - 5-minute tutorial
- **[Integration Guide](../docs/INTEGRATION_GUIDE.md)** - Add ACE to existing agents
- **[API Reference](../docs/API_REFERENCE.md)** - Complete API
- **[Complete ACE Guide](../docs/COMPLETE_GUIDE_TO_ACE.md)** - Deep dive

## 🔧 Adapting Examples

Each example is documented for easy adaptation:

1. **Browser automation**: Copy [browser-use/TEMPLATE.py](browser-use/TEMPLATE.py)
2. **LangChain**: See [langchain/README.md](langchain/README.md) patterns
3. **Custom agent**: Follow [custom_integration_example.py](custom_integration_example.py)
4. **Other**: Check subdirectory READMEs for guidance

## ❓ Need Help?

- [GitHub Issues](https://github.com/erwinh22/ACE/issues)
- Check subdirectory READMEs for specific guidance

Copyright (c) 2024 Kayba.ai
Copyright (c) 2025 RocketRag.ai (unified memory architecture, enterprise features, and enhancements)
Copyright (c) 2025 Levalo.ai (MCP integration, prompt enhancer, code retrieval optimization, adaptive chunking, enterprise features, and comprehensive enhancements)
