# 📚 ACE Framework API Reference

Complete API documentation for the ACE Framework.

## Configuration

### Centralized Configuration (ace/config.py)

**All embedding and Qdrant configuration is centralized** in `ace/config.py` using dataclasses with environment variable support.

#### EmbeddingConfig

```python
from ace.config import EmbeddingConfig

# Use defaults
config = EmbeddingConfig()
# config.url = "http://localhost:1234"
# config.model = "qwen/qwen3-embedding-8b"
# config.dimension = 4096

# Override with environment variables
# ACE_EMBEDDING_URL=http://localhost:1234
# ACE_EMBEDDING_MODEL=custom-model
# ACE_EMBEDDING_DIM=768

# Override programmatically
config = EmbeddingConfig(
    url="http://custom-server:1234",
    model="custom-embedding-model",
    dimension=768
)
```

**Environment Variables:**
- `ACE_EMBEDDING_URL` - Embedding server URL (default: `http://localhost:1234`)
- `ACE_EMBEDDING_MODEL` - Model name (default: `qwen/qwen3-embedding-8b`)
- `ACE_EMBEDDING_DIM` - Embedding dimension (default: `4096`)

#### QdrantConfig

```python
from ace.config import QdrantConfig

# Use defaults
config = QdrantConfig()
# config.url = "http://localhost:6333"
# config.unified_collection = "ace_memories_hybrid"

# Override with environment variables
# ACE_QDRANT_URL=http://qdrant-server:6333
# ACE_UNIFIED_COLLECTION=my_collection

# Override programmatically
config = QdrantConfig(
    url="http://qdrant-server:6333",
    collection_name="custom_collection"
)
```

**Environment Variables:**
- `ACE_QDRANT_URL` - Qdrant server URL (default: `http://localhost:6333`)
- `ACE_UNIFIED_COLLECTION` - Unified collection name (default: `ace_memories_hybrid`)
- `ACE_MEMORIES_COLLECTION` - Memories collection name (default: `ace_memories_hybrid`)

#### Using Configs with ACE Components

```python
from ace.config import EmbeddingConfig, QdrantConfig
from ace.unified_memory import UnifiedMemoryIndex

# Use defaults
index = UnifiedMemoryIndex()

# Inject custom configuration
embedding_config = EmbeddingConfig(url="http://custom:1234")
qdrant_config = QdrantConfig(collection_name="custom_collection")

index = UnifiedMemoryIndex(
    embedding_config=embedding_config,
    qdrant_config=qdrant_config
)
```

**Configuration-Aware Components:**
- `UnifiedMemoryIndex` - Unified memory indexing
- `SmartBulletIndex` - Bullet indexing and hybrid search
- `HydeRetrieval` - HyDE retrieval
- `DeduplicationService` - Memory deduplication
- `OptimizedRetrieval` - Optimized hybrid retrieval

**Benefits:**
- ✅ **Single Source of Truth** - All config in one file
- ✅ **Environment Variable Support** - Production vs development configs
- ✅ **Type Safety** - dataclass validation
- ✅ **IDE Support** - Auto-completion and type hints
- ✅ **Testability** - Easy to inject mock configs
- ✅ **Consistency** - Impossible to have conflicting defaults

---

## Unified Memory System

### UnifiedMemoryIndex

The `UnifiedMemoryIndex` provides a **single unified storage system** combining:
1. **ACE Framework Playbook bullets** (task strategies with helpful/harmful counters)
2. **Personal Memory Bank memories** (user preferences with severity/reinforcement)

Uses a single Qdrant collection with namespace separation for organization.

#### Quick Start

```python
from ace.unified_memory import UnifiedMemoryIndex, UnifiedBullet, UnifiedNamespace, UnifiedSource

# Create index (uses centralized config defaults)
index = UnifiedMemoryIndex()

# Create a user preference bullet
bullet = UnifiedBullet(
    id="pref-001",
    namespace=UnifiedNamespace.USER_PREFS,
    source=UnifiedSource.USER_FEEDBACK,
    content="User prefers TypeScript over JavaScript for new projects",
    section="preferences",
    severity=8,
    reinforcement_count=1
)

# Store the bullet (automatic deduplication)
result = index.index_bullet(bullet)
# Returns: {"stored": True, "action": "inserted", "similarity": 0.0, ...}

# Retrieve relevant bullets
results = index.retrieve(
    query="Which language should I use for this new project?",
    namespace=UnifiedNamespace.USER_PREFS,
    limit=5
)
# Returns list of UnifiedBullet objects, ranked by relevance
```

#### Namespaces

Organize bullets by type:

```python
from ace.unified_memory import UnifiedNamespace

# User preferences and workflow patterns
UnifiedNamespace.USER_PREFS

# Task execution strategies (from ACE playbook)
UnifiedNamespace.TASK_STRATEGIES

# Project-specific learnings
UnifiedNamespace.PROJECT_SPECIFIC
```

#### Sources

Track bullet origin:

```python
from ace.unified_memory import UnifiedSource

UnifiedSource.USER_FEEDBACK   # From explicit user input
UnifiedSource.TASK_EXECUTION   # Learned during task execution
UnifiedSource.MIGRATION        # Migrated from legacy system
UnifiedSource.EXPLICIT_STORE   # Manually stored
```

#### UnifiedBullet Schema

```python
@dataclass
class UnifiedBullet:
    # Identity
    id: str                              # Unique identifier
    namespace: UnifiedNamespace          # user_prefs | task_strategies | project_specific
    source: UnifiedSource                # Origin tracking
    
    # Content
    content: str                         # The actual strategy/lesson text
    section: str                         # Category (task_guidance, preferences, etc.)
    
    # ACE Scoring (for task strategies)
    helpful_count: int = 0               # Times this strategy helped
    harmful_count: int = 0               # Times this strategy hurt
    
    # Personal Memory Scoring (for user preferences)
    severity: int = 5                    # Importance level 1-10
    reinforcement_count: int = 0         # Times reinforced
    
    # Retrieval Optimization
    trigger_patterns: List[str] = []     # Patterns that suggest this bullet
    task_types: List[str] = []           # Types of tasks this applies to
    domains: List[str] = []              # Relevant domains
    complexity: str = "medium"           # simple | medium | complex
    
    # Timestamps
    created_at: datetime                 # When created
    last_reinforced: Optional[datetime]  # Last reinforcement
```

#### Deduplication

Automatic semantic deduplication prevents duplicate memories:

```python
# First storage
bullet1 = UnifiedBullet(
    id="test-001",
    namespace=UnifiedNamespace.USER_PREFS,
    source=UnifiedSource.USER_FEEDBACK,
    content="User prefers detailed error messages",
    severity=7
)
result1 = index.index_bullet(bullet1)
# {"stored": True, "action": "inserted", ...}

# Similar bullet - automatically reinforced instead of duplicated
bullet2 = UnifiedBullet(
    id="test-002",
    namespace=UnifiedNamespace.USER_PREFS,
    source=UnifiedSource.USER_FEEDBACK,
    content="Provide comprehensive error messages with stack traces",
    severity=8
)
result2 = index.index_bullet(bullet2)
# {"stored": False, "action": "reinforced", "similarity": 0.94, 
#  "existing_id": "test-001", "reinforcement_count": 2}
```

**Deduplication Parameters:**
- `enable_dedup`: Enable/disable deduplication (default: `True`)
- `dedup_threshold`: Similarity threshold for dedup (default: `0.92`)

#### Advanced Configuration

```python
from ace.config import EmbeddingConfig, QdrantConfig

# Custom configuration
embedding_config = EmbeddingConfig(
    url="http://custom-server:1234",
    model="qwen/qwen3-embedding-8b",
    dimension=4096
)

qdrant_config = QdrantConfig(
    url="http://qdrant-prod:6333",
    collection_name="production_unified"
)

# Inject into index
index = UnifiedMemoryIndex(
    embedding_config=embedding_config,
    qdrant_config=qdrant_config,
    enable_dedup=True,
    dedup_threshold=0.95  # Stricter dedup
)
```

#### Retrieval Modes

```python
# Semantic-only retrieval
results = index.retrieve(
    query="error handling patterns",
    namespace=UnifiedNamespace.TASK_STRATEGIES,
    limit=10
)

# Cross-namespace retrieval (no namespace filter)
results = index.retrieve(
    query="debugging approach",
    limit=5
)

# Namespace-specific with ACE scoring boost
results = index.retrieve(
    query="optimization techniques",
    namespace=UnifiedNamespace.TASK_STRATEGIES,
    limit=5
)
# Task strategies with high helpful_count are boosted
```

#### Collection Management

```python
# Initialize collection (idempotent)
index.initialize_collection()

# Check if collection exists
exists = index._collection_exists()

# Get all bullets from namespace
all_prefs = index.get_all_bullets(namespace=UnifiedNamespace.USER_PREFS)

# Delete specific bullet
index.delete_bullet(bullet_id="test-001")
```

---

## Core Components

### Generator

The Generator produces answers using the current playbook of strategies.

```python
from ace import Generator, LiteLLMClient

client = LiteLLMClient(model="gpt-4")
generator = Generator(client)

output = generator.generate(
    question="What is 2+2?",
    context="Show your work",
    playbook=playbook,
    reflection=None  # Optional reflection from previous attempt
)

# Output contains:
# - output.final_answer: The generated answer
# - output.reasoning: Step-by-step reasoning
# - output.bullet_ids: List of playbook strategies used
```

### Reflector

The Reflector analyzes what went right or wrong and tags which strategies helped or hurt.

```python
from ace import Reflector

reflector = Reflector(client)

reflection = reflector.reflect(
    question="What is 2+2?",
    generator_output=output,
    playbook=playbook,
    ground_truth="4",
    feedback="Correct!",
    max_refinement_rounds=1
)

# Reflection contains:
# - reflection.reasoning: Analysis of the outcome
# - reflection.error_identification: What went wrong (if anything)
# - reflection.root_cause_analysis: Why it went wrong
# - reflection.correct_approach: What should have been done
# - reflection.key_insight: Main lesson learned
# - reflection.bullet_tags: List of (bullet_id, tag) pairs
```

### Curator

The Curator transforms reflections into playbook updates.

```python
from ace import Curator

curator = Curator(client)

curator_output = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context="Math problems",
    progress="3/5 correct"
)

# Apply the updates
playbook.apply_delta(curator_output.delta)
```

## Playbook Management

### Creating a Playbook

```python
from ace import Playbook

playbook = Playbook()

# Add a strategy
bullet = playbook.add_bullet(
    section="Math Strategies",
    content="Break complex problems into smaller steps",
    metadata={"helpful": 5, "harmful": 0, "neutral": 1}
)
```

### Saving and Loading

```python
# Save to file
playbook.save_to_file("my_strategies.json")

# Load from file
loaded_playbook = Playbook.load_from_file("my_strategies.json")
```

### Playbook Statistics

```python
stats = playbook.stats()
# Returns:
# {
#   "sections": 3,
#   "bullets": 15,
#   "tags": {
#     "helpful": 45,
#     "harmful": 5,
#     "neutral": 10
#   }
# }
```

## Adapters

### OfflineAdapter

Train on a batch of samples.

```python
from ace import OfflineAdapter
from ace.types import Sample

adapter = OfflineAdapter(generator, reflector, curator)

samples = [
    Sample(
        question="What is 2+2?",
        context="Calculate",
        ground_truth="4"
    ),
    # More samples...
]

results = adapter.run(
    samples=samples,
    environment=environment,
    epochs=3,
    verbose=True
)
```

### OnlineAdapter

Learn from tasks one at a time.

```python
from ace import OnlineAdapter

adapter = OnlineAdapter(
    playbook=existing_playbook,
    generator=generator,
    reflector=reflector,
    curator=curator
)

for task in tasks:
    result = adapter.process(task, environment)
    # Playbook updates automatically after each task
```

## Integrations

ACE provides ready-to-use integrations with popular agentic frameworks. These classes wrap external agents with ACE learning capabilities.

### ACELiteLLM

Quick-start integration for simple conversational agents.

```python
from ace import ACELiteLLM

# Create an ACE-powered conversational agent
agent = ACELiteLLM(model="gpt-4o-mini")

# Ask questions - agent learns from each interaction
answer1 = agent.ask("What is the capital of France?")
answer2 = agent.ask("What about Spain?")

# Save learned strategies
agent.playbook.save_to_file("learned_strategies.json")

# Load and continue learning
agent2 = ACELiteLLM.from_playbook("learned_strategies.json", model="gpt-4o-mini")
```

**Parameters:**
- `model`: LiteLLM model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet")
- `playbook`: Optional existing Playbook to start with
- `ace_model`: Model for Reflector/Curator (defaults to same as main model)
- `**llm_kwargs`: Additional arguments passed to LiteLLMClient

### ACEAgent (browser-use)

Self-improving browser automation agent.

```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

# Create browser agent
llm = ChatBrowserUse(model="gpt-4o")
agent = ACEAgent(llm=llm)

# Run browser tasks - learns from successes and failures
await agent.run(task="Find the top post on Hacker News")
await agent.run(task="Search for ACE framework on GitHub")

# Playbook improves with each task
print(f"Learned {len(agent.playbook.bullets())} strategies")
```

**Parameters:**
- `llm`: Browser-use ChatBrowserUse instance
- `playbook`: Optional existing Playbook
- `ace_model`: Model for learning (defaults to "gpt-4o-mini")

**Requires:** `pip install browser-use` (optional dependency)

### ACELangChain

Wrap LangChain chains and agents with ACE learning.

```python
from ace import ACELangChain
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Create LangChain chain
llm = ChatOpenAI(temperature=0)
prompt = PromptTemplate.from_template("Answer this question: {question}")
chain = LLMChain(llm=llm, prompt=prompt)

# Wrap with ACE
ace_chain = ACELangChain(runnable=chain)

# Use like normal LangChain - but with learning!
result1 = ace_chain.invoke({"question": "What is 2+2?"})
result2 = ace_chain.invoke({"question": "What is 10*5?"})

# Access learned playbook
ace_chain.save_playbook("langchain_learned.json")
```

**Parameters:**
- `runnable`: Any LangChain Runnable (chains, agents, etc.)
- `playbook`: Optional existing Playbook
- `ace_model`: Model for learning (defaults to "gpt-4o-mini")
- `environment`: Custom evaluation environment (optional)

**Requires:** `pip install ace-framework[langchain]`

**See also:** [Integration Guide](INTEGRATION_GUIDE.md) for advanced patterns and custom integrations.

---

## Environments

### Creating Environments

All environments should extend the `TaskEnvironment` base class.

#### Simple Environment Example

Basic environment that compares output to ground truth using substring matching:

```python
from ace import TaskEnvironment, EnvironmentResult

class SimpleEnvironment(TaskEnvironment):
    """Basic environment for testing - checks if ground truth appears in answer."""

    def evaluate(self, sample, generator_output):
        # Simple substring matching (case-insensitive)
        correct = str(sample.ground_truth).lower() in str(generator_output.final_answer).lower()

        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )

# Usage
env = SimpleEnvironment()
result = env.evaluate(sample, generator_output)
```

### Custom Environments

```python
from ace import TaskEnvironment, EnvironmentResult

class CodeEnvironment(TaskEnvironment):
    def evaluate(self, sample, output):
        # Run the code
        success = execute_code(output.final_answer)

        return EnvironmentResult(
            feedback="Tests passed" if success else "Tests failed",
            ground_truth=sample.ground_truth,
            metrics={"pass_rate": 1.0 if success else 0.0}
        )
```

## LLM Clients

### LiteLLMClient

Support for 100+ LLM providers.

```python
from ace import LiteLLMClient

# Basic usage
client = LiteLLMClient(model="gpt-4")

# With configuration
client = LiteLLMClient(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    fallbacks=["claude-3-haiku", "gpt-3.5-turbo"]
)

# Generate completion
response = client.complete("What is the meaning of life?")
print(response.text)
```

### LangChainLiteLLMClient

Integration with LangChain.

```python
from ace.llm_providers import LangChainLiteLLMClient

client = LangChainLiteLLMClient(
    model="gpt-4",
    tags=["production"],
    metadata={"user": "alice"}
)
```

## Types

### Sample

```python
from ace.types import Sample

sample = Sample(
    question="Your question here",
    context="Optional context or requirements",
    ground_truth="Expected answer (optional)"
)
```

### GeneratorOutput

```python
@dataclass
class GeneratorOutput:
    reasoning: str
    final_answer: str
    bullet_ids: List[str]
    raw: Dict[str, Any]
```

### ReflectorOutput

```python
@dataclass
class ReflectorOutput:
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag]
    raw: Dict[str, Any]
```

### EnvironmentResult

```python
@dataclass
class EnvironmentResult:
    feedback: str
    ground_truth: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
```

## Delta Operations

### DeltaOperation Types

- `ADD`: Add new bullet to playbook
- `UPDATE`: Update existing bullet content
- `TAG`: Update helpful/harmful/neutral counts
- `REMOVE`: Remove bullet from playbook

```python
from ace.delta import DeltaOperation

op = DeltaOperation(
    type="ADD",
    section="Math Strategies",
    content="Always check your work",
    bullet_id="math-00001"
)
```

## Enterprise Modules

### Vector Search (`ace.qdrant_retrieval`)

Qdrant-based hybrid retrieval with dense embeddings and BM25 sparse vectors.

```python
from ace.qdrant_retrieval import QdrantBulletIndex, QdrantScoredResult

index = QdrantBulletIndex(
    qdrant_url="http://localhost:6333",
    embedding_url="http://localhost:1234",
    collection_name="ace_bullets",
    embedding_model="text-embedding-snowflake-arctic-embed-m-v1.5"
)

# Index playbook
index.index_playbook(playbook)

# Retrieve with hybrid search
results: List[QdrantScoredResult] = index.retrieve("query", limit=10)
```

### Async Retrieval (`ace.async_retrieval`)

Non-blocking async operations with parallel batch processing.

```python
from ace.async_retrieval import AsyncQdrantBulletIndex

async with AsyncQdrantBulletIndex() as index:
    # Single embedding
    embedding = await index.get_embedding("text")

    # Batch embeddings (parallel via asyncio.gather)
    embeddings = await index.batch_get_embeddings(["q1", "q2", "q3"])

    # Async retrieval
    results = await index.retrieve("query", limit=5)
```

### Caching (`ace.retrieval_caching`)

LRU caching with TTL expiration for embeddings and query results.

```python
from ace.retrieval_caching import EmbeddingCache, QueryResultCache

# Embedding cache (text -> vector)
emb_cache = EmbeddingCache(max_size=10000, ttl_seconds=3600)
emb_cache.put("text", [0.1] * 768)
embedding = emb_cache.get("text")  # Cache hit

# Query result cache with bullet-aware invalidation
query_cache = QueryResultCache(max_size=1000, ttl_seconds=600)
query_cache.put("query", results)
query_cache.invalidate_bullet("bullet_id")  # Invalidates related queries

# Metrics
print(f"Hit rate: {emb_cache.hit_rate():.2%}")
```

### Horizontal Scaling (`ace.scaling`)

Sharded collections and clustered Qdrant with load balancing.

```python
from ace.scaling import (
    ShardedBulletIndex,
    QdrantCluster,
    ShardStrategy,
    LoadBalancingStrategy,
    ClusterHealthCheck
)

# Sharded by tenant/domain/hybrid
sharded = ShardedBulletIndex(
    qdrant_client=client,
    shard_strategy=ShardStrategy.TENANT
)
sharded.index_bullet(bullet, tenant_id="acme_corp")
results = sharded.retrieve("query", tenant_id="acme_corp")

# Clustered Qdrant
cluster = QdrantCluster(
    nodes=["http://node1:6333", "http://node2:6333"],
    strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
    max_consecutive_failures=3
)
results = cluster.retrieve("query")

# Health monitoring
health_checker = ClusterHealthCheck(cluster)
status = health_checker.check_all_nodes()
```

### Code Analysis (`ace.code_analysis`)

Tree-sitter based multi-language AST parsing.

```python
from ace.code_analysis import CodeAnalyzer, CodeSymbol

analyzer = CodeAnalyzer()

# Parse code
symbols: List[CodeSymbol] = analyzer.extract_symbols(code, "python")
for s in symbols:
    print(f"{s.kind}: {s.name} (lines {s.start_line}-{s.end_line})")

# Find specific symbol
symbol = analyzer.find_symbol(code, "MyClass.method", "python")
body = analyzer.get_symbol_body(code, "my_function", "python")

# Supported languages: python, typescript, javascript, go
```

### Dependency Graph (`ace.dependency_graph`)

Import and call graph analysis.

```python
from ace.dependency_graph import DependencyGraph

graph = DependencyGraph()

# Extract imports
imports = graph.extract_imports(code, "python")

# Build call graph
call_edges = graph.build_call_graph(code, "python")

# Find callers/callees
callers = graph.find_callers(code, "my_function", "python")
callees = graph.find_callees(code, "my_function", "python")
```

### Security (`ace.security`)

Authentication and authorization.

```python
from ace.security import (
    APIKeyAuth,
    JWTAuth,
    RoleBasedAccessControl,
    SecurityMiddleware
)

# API Key authentication
api_auth = APIKeyAuth(valid_keys={"sk-xxx": "user123"})
is_valid = api_auth.validate("sk-xxx")

# JWT authentication
jwt_auth = JWTAuth(secret_key="your-secret")
token = jwt_auth.create_token(user_id="user123", roles=["editor"])
user_context = jwt_auth.validate_token(token)

# RBAC
rbac = RoleBasedAccessControl()
rbac.grant_permission("user123", "playbook:write", "my_playbook")
can_write = rbac.check_permission("user123", "playbook:write", "my_playbook")

# Security middleware
middleware = SecurityMiddleware(auth_method="jwt", jwt_auth=jwt_auth, rbac=rbac)
```

### Multi-Tenancy (`ace.multitenancy`)

Tenant isolation for playbooks and collections.

```python
from ace.multitenancy import TenantContext, TenantManager

manager = TenantManager(base_path="./tenant_data")

# Tenant-scoped operations
with TenantContext(tenant_id="tenant-a"):
    manager.save_playbook(playbook, "my_playbook")
    loaded = manager.load_playbook("my_playbook")  # Isolated to tenant-a

# Cross-tenant access prevented
with TenantContext(tenant_id="tenant-b"):
    loaded = manager.load_playbook("my_playbook")  # Different playbook
```

### Audit Logging (`ace.audit`)

Enterprise audit logging with JSONL persistence.

```python
from ace.audit import AuditLogger

logger = AuditLogger(log_dir="./audit_logs")

# Log operations
logger.log_retrieval(
    query="search query",
    results=results,
    user_id="user123",
    tenant_id="tenant-a"
)

# Export logs
logger.export_json("2025-12-01", "2025-12-31", output_file="audit.json")
logger.export_csv("2025-12-01", "2025-12-31", output_file="audit.csv")

# Get metrics
metrics = logger.get_metrics("2025-12-01", "2025-12-31")
print(f"Total queries: {metrics['query_count']}, Avg latency: {metrics['avg_latency_ms']}ms")
```

### Observability (`ace.observability`)

Production monitoring with Prometheus, health checks, and tracing.

```python
from ace.observability.metrics import (
    track_latency,
    track_operation,
    retrieval_latency_histogram,
    retrieval_count
)
from ace.observability.health import HealthChecker
from ace.observability.tracing import TracingManager, trace_operation

# Track latency
with track_latency(operation="semantic_search", tenant_id="tenant-123"):
    results = perform_search(query)

# Health checks
checker = HealthChecker(
    qdrant_url="http://localhost:6333",
    embedding_url="http://localhost:1234"
)
status = checker.check_all()  # {"qdrant": {"healthy": True, "latency_ms": 5.2}, ...}

# Distributed tracing (requires OpenTelemetry)
tracing = TracingManager(service_name="ace-service")

@trace_operation("retrieval")
def retrieve_bullets(query):
    return index.retrieve(query)
```

## Prompts

### Using Default Prompts

```python
from ace.prompts import GENERATOR_PROMPT, REFLECTOR_PROMPT, CURATOR_PROMPT

generator = Generator(client, prompt_template=GENERATOR_PROMPT)
```

### Using v2.1 Prompts (Recommended)

ACE v2.1 prompts show +17% success rate improvement vs v1.0.

```python
from ace.prompts_v2_1 import PromptManager

manager = PromptManager(default_version="2.1")

generator = Generator(
    client,
    prompt_template=manager.get_generator_prompt(domain="math")
)
```

**Note:** v2.0 prompts (`ace.prompts_v2`) are deprecated. Use v2.1 for best performance.

### Custom Prompts

```python
custom_prompt = '''
Playbook: {playbook}
Question: {question}
Context: {context}

Generate a JSON response with:
- reasoning: Your step-by-step thought process
- bullet_ids: List of playbook IDs you used
- final_answer: Your answer
'''

generator = Generator(client, prompt_template=custom_prompt)
```

## Async Operations

```python
import asyncio

async def main():
    # Async completion
    response = await client.acomplete("What is 2+2?")

    # Async adapter operations also supported
    # (Implementation depends on adapter async support)

asyncio.run(main())
```

## Streaming

```python
# Stream responses token by token
for chunk in client.complete_with_stream("Write a story"):
    print(chunk, end="", flush=True)
```

## Error Handling

```python
from ace.exceptions import ACEException

try:
    output = generator.generate(...)
except ACEException as e:
    print(f"ACE error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Google
export GOOGLE_API_KEY="your-key"

# Custom endpoint
export LITELLM_API_BASE="https://your-endpoint.com"
```

### Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or just for ACE
logging.getLogger("ace").setLevel(logging.DEBUG)
```

## Best Practices

1. **Start with SimpleEnvironment**: Get basic training working first
2. **Use fallback models**: Ensure reliability in production
3. **Save playbooks regularly**: Preserve learned strategies
4. **Monitor costs**: Track token usage with metrics
5. **Test with dummy mode**: Validate logic without API calls
6. **Use appropriate epochs**: 2-3 epochs usually sufficient
7. **Implement custom environments**: Tailor evaluation to your task

## Examples

See the [examples](../examples/) directory for complete working examples:

**Core Examples:**
- `simple_ace_example.py` - Basic usage
- `playbook_persistence.py` - Save/load strategies

**By Category:**
- [starter-templates/](../examples/starter-templates/) - Quick start templates
- [langchain/](../examples/langchain/) - LangChain integration examples
- [prompts/](../examples/prompts/) - Prompt engineering examples
- [browser-use/](../examples/browser-use/) - Browser automation