# ACE Prompt Enhancer Tool

## Overview

The `ace_enhance_prompt` tool transforms vague, incomplete user prompts into comprehensive, structured, actionable specifications optimized for AI agent execution. It uses an LLM with multi-source context injection to create detailed prompts that eliminate ambiguity and reduce follow-up questions.

**Location**: `ace_mcp_server.py:1161-1368`
**System Prompt**: `ace/prompts/enhance_prompt.md`

---

## Quick Start

### Basic Usage

```json
{
  "prompt": "add oauth support"
}
```

**Returns**:
```markdown
## OBJECTIVE
Implement OAuth 2.0 authentication support in the API client...

## CONTEXT
- Current State: ApiClient uses basic authentication...
- Reference Implementation: AdminClient already has OAuth...

## REQUIREMENTS
### Functional Requirements
- Implement OAuth 2.0 authorization code flow with PKCE
- Support token refresh automatically...
...
```

### With Full Context

```json
{
  "prompt": "fix the login bug",
  "include_memories": true,
  "include_git_commits": true,
  "include_git_status": true,
  "open_files": "src/auth/login.ts, src/types/user.ts",
  "chat_history": "User: The login fails after password reset...",
  "workspace_path": "/path/to/project",
  "provider": "openai",
  "model": "gpt-4o"
}
```

---

## Tool Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | The original user prompt to enhance |
| `include_memories` | boolean | `true` | Include relevant ACE memories as context |
| `include_git_commits` | boolean | `false` | Include recent git commit history |
| `include_git_status` | boolean | `false` | Include current git status (staged/unstaged changes) |
| `open_files` | string | `""` | List of file paths currently open in editor |
| `chat_history` | string | `""` | Recent conversation history for context |
| `custom_context` | string | `""` | Any additional context to inject |
| `workspace_path` | string | `""` | Workspace path for git operations (uses cwd if empty) |
| `provider` | string | `"zai"` | LLM provider: `zai`, `openai`, `anthropic`, `lmstudio` |
| `model` | string | `""` | Specific model override (empty = provider default) |
| `max_tokens` | integer | `8000` | Maximum response tokens |
| `temperature` | float | `0.3` | Response creativity (0.0-1.0) |

---

## Context Sources

| Source | Description | Example Content |
|--------|-------------|-----------------|
| **ACE Memories** | Relevant preferences, patterns, lessons learned | User coding style, project conventions |
| **Git Commits** | Recent commit history (last 5) | `5decf23 feat(mcp): add ace_enhance_prompt tool` |
| **Git Status** | Current modified/staged files | `M ace_mcp_server.py` |
| **Open Files** | Files currently in editor | File paths or content |
| **Chat History** | Recent conversation messages | User: "The login fails..." |
| **Custom Context** | Any additional information | User-provided notes |

---

## LLM Provider Configuration

### Supported Providers

| Provider | Default Model | API Endpoint | Environment Variable |
|----------|---------------|--------------|----------------------|
| `zai` | `glm-4.7` | `https://api.z.ai/v1` | `ZAI_API_KEY` |
| `openai` | `gpt-4o` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | `claude-sonnet-4-20250514` | `https://api.anthropic.com/v1` | `ANTHROPIC_API_KEY` |
| `lmstudio` | *required in `model` param* | `http://localhost:1234/v1` | None (local) |

### Provider Selection Examples

```json
// Use Z.AI (default)
{"provider": "zai"}

// Use OpenAI with GPT-4
{"provider": "openai", "model": "gpt-4o"}

// Use Anthropic Claude
{"provider": "anthropic", "model": "claude-sonnet-4-20250514"}

// Use local LM Studio
{"provider": "lmstudio", "model": "your-model-name"}
```

---

## Enhancement Output Structure

The enhancer produces different structures based on intent classification:

### For Analytical Queries

**Keywords**: `compare`, `analyze`, `understand`, `investigate`, `evaluate`, `assess`

```markdown
## OBJECTIVE
[Clear goal with conversational context]

## ANALYSIS APPROACH
1. **[First Area]**: What to examine
   - Specific aspects to identify with file references
   - Where to look: exact paths and line numbers

2. **[Second Area]**: Comparison or evaluation focus
   - Specific comparison points
   - What to identify: gaps, patterns, discrepancies

Please provide a structured [format] showing:
- Output element 1 with code references
- Gaps, differences, or discrepancies identified
- Insights about strategy or architecture
```

**Characteristics**:
- Conversational and readable
- Minimal checkboxes
- No CONSTRAINTS or DEFINITION OF DONE sections
- Focus on understanding and insights

### For Implementation Queries

**Keywords**: `create`, `implement`, `build`, `fix`, `modify`, `refactor`

```markdown
## OBJECTIVE
[Action Verb] [Target] into [Result] [Using Pattern]

## CONTEXT
- Current State: [What exists with file paths]
- Current Location: [Absolute path to code]
- Reference Implementation: [Similar implementations]
- Gap: [What's missing]

## REQUIREMENTS
### Functional Requirements
- [Specific action with exact signature]

### Non-Functional Requirements
- [Performance requirement with metrics]

## IMPLEMENTATION APPROACH
**Strategy**: [High-level approach]

**Steps**:
1. **[Phase Name]** (`path/to/file.ts`):
   - [Specific action with implementation detail]

## CONSTRAINTS
- Do not modify [component with reason]

## VERIFICATION STEPS
1. **Test [Specific Aspect]**:
   ```bash
   [Exact command with expected output]
   ```

## DEFINITION OF DONE
- [Executable verification with command]
- [Measurable outcome]
```

**Characteristics**:
- Structured template with all sections
- Checkboxes for tracking
- Detailed verification commands
- Architecture diagrams

---

## Enhancement Methodology

The tool follows a four-phase enhancement process:

### Phase 1: Intent Analysis
- Parse literal request
- Identify implied intent
- Detect ambiguities
- Classify as ANALYTICAL or IMPLEMENTATION

### Phase 2: Risk Assessment
- Scope: File | Module | System-wide
- Risk: Low | Medium | High
- Complexity: Simple | Standard | Complex

### Phase 3: Context Gathering
Based on intent and risk:
- **DEBUG**: Current implementation, error patterns, recent changes
- **CREATE**: Architectural patterns, reference implementations, integration points
- **MODIFY**: Current implementation, dependencies, breaking change impact
- **REFACTOR**: Current implementation, all references, test suite
- **ANALYTICAL**: Full system context, specifications, comparison targets

### Phase 4: Structure Selection
Choose appropriate template based on intent classification

---

## Enhancement Quality Rules

### Critical Rules (Enforced)

1. **Always Include Concrete Details**
   - Bad: "Create an MCP server"
   - Good: "Create MCP server entry point (`src/mcp/server.ts`) that exposes context engine via stdio transport using JSON-RPC 2.0"

2. **Always Provide Reference Implementations**
   - Bad: "Follow existing patterns"
   - Good: "Follow the pattern in `docs/xyz.md` lines 1001-1163"

3. **Always Specify Exact Technologies**
   - Bad: "Use appropriate communication protocol"
   - Good: "Use stdio JSON-RPC 2.0 protocol as specified in MCP specification"

4. **Never Ask Follow-Up Questions**
   - Bad: "Which file needs error handling?"
   - Good: "Add error handling to `src/services/api-client.ts`"

### Heuristics Applied

- **KISS**: Challenge complexity, remove unnecessary abstractions
- **DRY**: Reference existing patterns, reuse components
- **Fail-Fast**: Add validation at entry points, specify error conditions
- **Surgical Precision**: Specify exactly what to change, preserve everything else

---

## Examples

### Example 1: Vague Request Enhanced

**Input**: `"add oauth support"`

**Output** (structured prompt with):
- OBJECTIVE: Implement OAuth 2.0 authentication
- CONTEXT: Current ApiClient uses basic auth, reference to AdminClient OAuth mixin
- REQUIREMENTS: Authorization code flow, token refresh, secure storage
- IMPLEMENTATION APPROACH: Create mixin, update client, add tests
- VERIFICATION: Test commands with expected outputs
- DEFINITION OF DONE: Checkboxes for completion tracking

### Example 2: Analytical Query

**Input**: `"how does the MCP server context compare to docs?"`

**Output** (analytical structure):
- OBJECTIVE: Compare actual MCP context implementation against documented specification
- ANALYSIS APPROACH:
  1. Examine context elements in `src/context/contextPack.ts`
  2. Review `docs/context-specification.md` for documented elements
  3. Gap analysis comparing implementation vs documentation
- Request for comparison table showing gaps, discrepancies, and insights

---

## Integration with MCP Clients

### VS Code Copilot

```json
{
  "mcpServers": {
    "ace": {
      "command": "python",
      "args": ["path/to/ace_mcp_server.py"]
    }
  }
}
```

### Claude Desktop

```json
{
  "mcpServers": {
    "ace": {
      "command": "python",
      "args": ["/path/to/ace_mcp_server.py"]
    }
  }
}
```

### Usage in AI Agents

```python
# Call the tool before executing complex tasks
enhanced = await mcp_client.call_tool(
    "ace_enhance_prompt",
    arguments={
        "prompt": user_request,
        "include_memories": True,
        "include_git_commits": True
    }
)

# Review the enhanced prompt
print(enhanced)

# Then execute with the detailed specification
result = await agent.execute(enhanced)
```

---

## System Prompt

The enhancement system prompt is located at `ace/prompts/enhance_prompt.md` (1100 lines).

Key sections:
- Core mission and methodology
- Intent classification rules
- Context gathering strategies
- Structure selection criteria
- Quality checkpoints
- Concrete before/after examples
- Self-validation rules

---

## Troubleshooting

### No API Key Configured

**Error**: `Error: No API key configured for provider 'zai'`

**Solution**: Set the appropriate environment variable:
```bash
export ZAI_API_KEY="your-api-key"
# or
export OPENAI_API_KEY="your-api-key"
```

### Enhancement Timeout

**Error**: Timeout after 120 seconds

**Solution**: The tool has a 120-second timeout for complex enhancements. If exceeded:
- Use a faster model
- Reduce context (set `include_git_commits=false`)
- Simplify the original prompt

### Poor Quality Enhancement

**Symptom**: Enhanced prompt is still vague or generic

**Solutions**:
1. Enable more context sources (`include_memories=true`, `include_git_commits=true`)
2. Use a more capable model (`gpt-4o`, `claude-sonnet-4`)
3. Provide additional context via `custom_context` parameter

---

## Architecture

```
User Prompt (vague)
        |
        v
ace_enhance_prompt Tool
        |
        +-- Context Injection
        |   +-- ACE Memories (relevant patterns)
        |   +-- Git Commits (recent history)
        |   +-- Git Status (current changes)
        |   +-- Open Files (editor context)
        |   +-- Chat History (conversation)
        |   +-- Custom Context (user provided)
        |
        v
LLM Provider (zai/openai/anthropic/lmstudio)
        |
        v
Enhanced Prompt (structured, actionable)
```

---

## References

- **Source Code**: `ace_mcp_server.py:1161-1368`
- **System Prompt**: `ace/prompts/enhance_prompt.md`
- **MCP Integration**: `docs/MCP_INTEGRATION.md`
- **Research Paper**: [Agentic Context Engineering](https://arxiv.org/abs/2510.04618)

---

*Last Updated: 2025-01-06*
