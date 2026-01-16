"""Code indexer module for workspace code indexing.

This module provides code indexing capabilities that scan a workspace,
parse files using AdaptiveChunker, and store indexed chunks in Qdrant
for semantic search.

Configuration:
    ACE_CODE_COLLECTION: Qdrant collection name (default: ace_code_context)
    ACE_CODE_EMBEDDING_DIM: Embedding dimension (default: from EmbeddingConfig)
    QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
    ACE_ADAPTIVE_CHUNKING: Enable adaptive file-type-aware chunking (default: true)

The indexer supports:
- Multi-language code parsing (Python, JavaScript, TypeScript, Go) via ASTChunker
- Adaptive chunking for non-code files (Markdown, YAML, JSON, TOML)
- Section-based chunking for documentation (by headers)
- Structure-based chunking for config files (by keys/sections)
- Incremental updates on file changes
- File watching for auto-updates
- Gitignore and custom exclude pattern support
"""

from __future__ import annotations

import os
import logging
import hashlib
import fnmatch
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CodeChunkIndexed:
    """A code chunk ready for indexing with all metadata."""
    
    content: str
    file_path: str  # Relative to workspace root
    start_line: int
    end_line: int
    language: str
    symbols: List[str] = field(default_factory=list)
    chunk_hash: str = ""
    
    def __post_init__(self):
        if not self.chunk_hash:
            self.chunk_hash = hashlib.md5(
                f"{self.file_path}:{self.start_line}:{self.content}".encode()
            ).hexdigest()


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

# File extension to language mapping
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    # Documentation files (for comprehensive context like ThatOtherContextEngine MCP)
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    # Configuration files
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
}

# Supported code extensions for scanning
SUPPORTED_CODE_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys())

# Default exclude patterns
# Note: fnmatch ** only works for paths with prefix (e.g., foo/.venv)
# For paths starting with the pattern (e.g., .venv/...), we need direct patterns
DEFAULT_EXCLUDE_PATTERNS = [
    # Node.js
    "**/node_modules/**",
    "node_modules/**",
    "node_modules/*",
    # Git
    "**/.git/**",
    ".git/**",
    ".git/*",
    # Python cache
    "**/__pycache__/**",
    "__pycache__/**",
    "__pycache__/*",
    # Python virtual environments
    "**/.venv/**",
    ".venv/**",
    ".venv/*",
    "**/venv/**",
    "venv/**",
    "venv/*",
    "**/.env/**",
    ".env/**",
    "**/env/**",
    "env/**",
    # Build outputs
    "**/dist/**",
    "dist/**",
    "**/build/**",
    "build/**",
    # IDE folders
    "**/.idea/**",
    ".idea/**",
    "**/.vscode/**",
    ".vscode/**",
    # Test coverage
    "**/coverage/**",
    "coverage/**",
    "**/htmlcov/**",
    "htmlcov/**",
    # Minified files
    "**/*.min.js",
    "**/*.min.css",
    "**/*.map",
    # Package managers
    "**/vendor/**",
    "vendor/**",
    # Rust
    "**/target/**",
    "target/**",
    # .NET
    "**/bin/**",
    "bin/**",
    "**/obj/**",
    "obj/**",
    # Python eggs and packages
    "**/*.egg-info/**",
    "*.egg-info/**",
    "**/.eggs/**",
    ".eggs/**",
    # Pytest cache
    "**/.pytest_cache/**",
    ".pytest_cache/**",
    # mypy cache
    "**/.mypy_cache/**",
    ".mypy_cache/**",
    # Jupyter checkpoints
    "**/.ipynb_checkpoints/**",
    ".ipynb_checkpoints/**",
]


# =============================================================================
# CODE INDEXER
# =============================================================================

class CodeIndexer:
    """
    Index workspace files for semantic search.
    
    Scans workspace directories, parses files using AdaptiveChunker,
    generates embeddings, and stores in Qdrant for retrieval.
    
    Features:
    - Adaptive file-type-aware chunking (code, docs, config)
    - Multi-language code support via ASTChunker (Python, JS, TS, Go)
    - Section-based chunking for documentation (Markdown headers)
    - Structure-based chunking for config files (YAML, JSON, TOML)
    - Incremental updates on file changes
    - File watching for auto-updates
    - Gitignore and exclude pattern support
    - Relative path storage for portability
    
    Example:
        indexer = CodeIndexer(workspace_path="/my/project")
        stats = indexer.index_workspace()
        print(f"Indexed {stats['files_indexed']} files")
    """
    
    def __init__(
        self,
        workspace_path: str,
        qdrant_url: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        exclude_patterns: Optional[List[str]] = None,
        respect_gitignore: bool = False,
    ):
        """
        Initialize the code indexer.
        
        Args:
            workspace_path: Root directory of the workspace to index
            qdrant_url: Qdrant server URL (default: from env or localhost:6333)
            collection_name: Qdrant collection name (default: from env or ace_code_context)
            embedding_dim: Embedding dimension (default: 1024d for Voyage-code-3)
            embed_fn: Custom embedding function (default: None, uses Voyage code embedder)
            exclude_patterns: Additional patterns to exclude from scanning
            respect_gitignore: Whether to respect .gitignore files
        """
        # Load Voyage code embedding config (REQUIRED)
        from ace.config import VoyageCodeEmbeddingConfig
        
        _voyage_config = VoyageCodeEmbeddingConfig()
        
        if not _voyage_config.is_configured():
            raise RuntimeError(
                "VOYAGE_API_KEY environment variable is required for code indexing. "
                "Set VOYAGE_API_KEY to use voyage-code-3."
            )
        
        default_dim = _voyage_config.dimension  # 1024d for Voyage
        logger.info(f"Using Voyage config: {_voyage_config.model} ({default_dim}d)")
        
        self.workspace_path = os.path.abspath(workspace_path)
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name or os.environ.get("ACE_CODE_COLLECTION", "ace_code_context")
        self.embedding_dim = embedding_dim or int(os.environ.get("ACE_CODE_EMBEDDING_DIM", str(default_dim)))
        self._embed_fn = embed_fn
        self.respect_gitignore = respect_gitignore
        
        # Combine default and custom exclude patterns
        self._exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)
        if exclude_patterns:
            self._exclude_patterns.extend(exclude_patterns)
        
        # Load gitignore patterns
        self._gitignore_patterns: List[str] = []
        if self.respect_gitignore:
            self._load_gitignore()
        
        # Qdrant client
        self._client = None
        self._init_qdrant()
        
        # File watcher state
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop_event = threading.Event()
        self._watching = False
        
        # Track indexed files
        self._indexed_files: Set[str] = set()
    
    def _load_gitignore(self) -> None:
        """Load patterns from .gitignore file."""
        gitignore_path = Path(self.workspace_path) / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Convert gitignore pattern to glob pattern
                            pattern = line.strip("/")
                            if not pattern.startswith("**/"):
                                pattern = f"**/{pattern}"
                            self._gitignore_patterns.append(pattern)
            except Exception as e:
                logger.warning(f"Failed to load .gitignore: {e}")
    
    def _init_qdrant(self) -> None:
        """Initialize Qdrant client and collection with hybrid search support."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, SparseVectorParams
            
            self._client = QdrantClient(url=self.qdrant_url)
            
            # Create collection if not exists - with hybrid vectors (dense + sparse)
            if not self._client.collection_exists(self.collection_name):
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(),
                    },
                )
                logger.info(f"Created Qdrant collection with hybrid vectors: {self.collection_name}")
        except ImportError:
            logger.warning("qdrant-client not installed, indexing will be mocked")
            self._client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Qdrant: {e}")
            self._client = None
    
    def _get_embedder(self) -> Callable[[str], List[float]]:
        """Get or create code-specific embedding function.
        
        REQUIRES Voyage API (voyage-code-3, 1024d).
        Voyage-code-3 is specifically trained for code retrieval.
        
        Raises:
            RuntimeError: If VOYAGE_API_KEY is not configured
        """
        if self._embed_fn:
            return self._embed_fn
        
        # Voyage API is REQUIRED for code embeddings
        from ace.config import VoyageCodeEmbeddingConfig
        voyage_config = VoyageCodeEmbeddingConfig()
        
        if not voyage_config.is_configured():
            raise RuntimeError(
                "VOYAGE_API_KEY environment variable is required for code embeddings. "
                "Voyage-code-3 is the only supported code embedding model. "
                "Get your API key from https://www.voyageai.com/"
            )
        
        try:
            import voyageai
        except ImportError:
            raise RuntimeError(
                "voyageai package is required for code embeddings. "
                "Install with: pip install voyageai"
            )
        
        # Create Voyage client
        vo_client = voyageai.Client(api_key=voyage_config.api_key)
        logger.info(f"Using Voyage {voyage_config.model} for code indexing ({voyage_config.dimension}d)")
        
        # Update dimension for collection creation
        self.embedding_dim = voyage_config.dimension
        
        # Store client and config for batch operations
        self._voyage_client = vo_client
        self._voyage_config = voyage_config
        
        def voyage_embed(text: str) -> List[float]:
            """Embed text using Voyage code model for documents."""
            try:
                result = vo_client.embed(
                    [text],
                    model=voyage_config.model,
                    input_type=voyage_config.document_input_type
                )
                return result.embeddings[0]
            except Exception as e:
                logger.error(f"Voyage embedding error: {e}")
                return [0.0] * voyage_config.dimension
        
        self._embed_fn = voyage_embed
        return voyage_embed

    def _embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = None,  # Uses config or default 500
        max_tokens_per_batch: int = None,  # Uses config or default 115K
        max_concurrent: int = None  # Uses config or default 4
    ) -> List[List[float]]:
        """Batch embed texts using Voyage API with token-aware splitting and parallelization.
        
        Voyage API limits for voyage-code-3:
        1. Max 1,000 texts per request (we use 500 for safety)
        2. Max 120K tokens per request (we use 115K for safety)
        3. 2,000 RPM / 3M TPM at Tier 1
        
        This function respects all limits and parallelizes for speed.
        Configure via environment variables:
        - ACE_VOYAGE_BATCH_SIZE: texts per batch (default: 500)
        - ACE_VOYAGE_BATCH_TOKENS: max tokens per batch (default: 115000)
        - ACE_VOYAGE_PARALLEL: parallel batches (default: 4)
        
        Args:
            texts: List of texts to embed
            batch_size: Max texts per API call (default from config)
            max_tokens_per_batch: Max estimated tokens per batch (default from config)
            max_concurrent: Max parallel API calls (default from config)
            
        Returns:
            List of embedding vectors
        """
        import concurrent.futures
        
        # Ensure embedder is initialized
        self._get_embedder()
        
        # Use config values if not explicitly provided
        if hasattr(self, '_voyage_config') and self._voyage_config:
            batch_size = batch_size or self._voyage_config.batch_size
            max_tokens_per_batch = max_tokens_per_batch or self._voyage_config.batch_max_tokens
            max_concurrent = max_concurrent or self._voyage_config.parallel_batches
        else:
            batch_size = batch_size or 300
            max_tokens_per_batch = max_tokens_per_batch or 80000
            max_concurrent = max_concurrent or 4
        
        if not hasattr(self, '_voyage_client') or not self._voyage_client:
            # Fallback to individual calls
            embed_fn = self._get_embedder()
            return [embed_fn(t) for t in texts]
        
        # Build token-aware batches
        def estimate_tokens(text: str) -> int:
            """Estimate tokens - code uses ~2-3 chars per token due to symbols.
            
            Using 2 chars/token (conservative) to avoid exceeding Voyage's 120K limit.
            Error showed 178K actual tokens when we estimated 115K using 4 chars/token,
            meaning actual ratio was ~2.6 chars/token for code files.
            """
            return max(1, len(text) // 2)
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            text_tokens = estimate_tokens(text)
            
            # Check if adding this text would exceed limits
            if current_batch and (len(current_batch) >= batch_size or current_tokens + text_tokens > max_tokens_per_batch):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(text)
            current_tokens += text_tokens
        
        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Split {len(texts)} texts into {len(batches)} batches (batch_size={batch_size}, max_tokens={max_tokens_per_batch})")
        
        def embed_single_batch(batch_info):
            """Embed a single batch and return (index, embeddings)."""
            batch_num, batch = batch_info
            batch_tokens = sum(estimate_tokens(t) for t in batch)
            
            try:
                result = self._voyage_client.embed(
                    batch,
                    model=self._voyage_config.model,
                    input_type=self._voyage_config.document_input_type
                )
                logger.info(f"Batch {batch_num}/{len(batches)}: embedded {len(batch)} chunks (~{batch_tokens} tokens)")
                return (batch_num, result.embeddings)
            except Exception as e:
                logger.error(f"Batch {batch_num} embedding error: {e}")
                # Fallback: zero vectors for failed batch
                return (batch_num, [[0.0] * self._voyage_config.dimension] * len(batch))
        
        # Parallel execution for speed
        if len(batches) > 1 and max_concurrent > 1:
            all_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = {executor.submit(embed_single_batch, (i+1, b)): i for i, b in enumerate(batches)}
                for future in concurrent.futures.as_completed(futures):
                    all_results.append(future.result())
            
            # Sort by batch number to maintain order
            all_results.sort(key=lambda x: x[0])
            all_embeddings = []
            for _, embeddings in all_results:
                all_embeddings.extend(embeddings)
        else:
            # Sequential for single batch
            all_embeddings = []
            for batch_num, batch in enumerate(batches, 1):
                _, embeddings = embed_single_batch((batch_num, batch))
                all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def _should_exclude(self, file_path: str) -> bool:
        """Check if file should be excluded based on patterns."""
        # Skip Windows special devices (nul, con, prn, etc.)
        basename = os.path.basename(file_path).lower()
        if basename in ("nul", "con", "prn", "aux", "com1", "com2", "com3", "com4", "lpt1", "lpt2", "lpt3"):
            return True
        
        try:
            rel_path = os.path.relpath(file_path, self.workspace_path)
        except ValueError:
            # Path on different mount (e.g., \\.\nul vs D:\)
            return True
        rel_path = rel_path.replace("\\", "/")  # Normalize for pattern matching
        
        # Check exclude patterns
        for pattern in self._exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            if fnmatch.fnmatch(f"/{rel_path}", pattern):
                return True
        
        # Check gitignore patterns
        for pattern in self._gitignore_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            if fnmatch.fnmatch(f"/{rel_path}", pattern):
                return True
        
        return False
    
    def _is_code_file(self, file_path: str) -> bool:
        """Check if file is a supported code file."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in SUPPORTED_CODE_EXTENSIONS:
            return True
        
        # Handle extensionless files by filename
        filename = os.path.basename(file_path).upper()
        extensionless_files = {
            'LICENSE', 'LICENCE', 'MAKEFILE', 'DOCKERFILE', 'GEMFILE',
            'PROCFILE', 'RAKEFILE', 'VAGRANTFILE', 'BREWFILE',
            'CODEOWNERS', 'AUTHORS', 'CONTRIBUTORS', 'THANKS',
            'COPYING', 'PATENTS', 'NOTICE', 'CREDITS'
        }
        if filename in extensionless_files:
            return True
        
        return False
    
    def _is_binary_file(self, file_path: str) -> bool:
        """Check if file appears to be binary."""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(8192)
                # Check for null bytes (common in binary files)
                if b"\x00" in chunk:
                    return True
                # Check for high ratio of non-text bytes
                text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
                non_text = sum(1 for b in chunk if b not in text_chars)
                if non_text / len(chunk) > 0.3:
                    return True
        except Exception:
            return True
        return False
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext:
            return EXTENSION_TO_LANGUAGE.get(ext, "unknown")
        
        # Handle extensionless files
        filename = os.path.basename(file_path).upper()
        extensionless_languages = {
            'LICENSE': 'text', 'LICENCE': 'text', 'COPYING': 'text',
            'PATENTS': 'text', 'NOTICE': 'text', 'AUTHORS': 'text',
            'CONTRIBUTORS': 'text', 'THANKS': 'text', 'CREDITS': 'text',
            'MAKEFILE': 'makefile', 'DOCKERFILE': 'dockerfile',
            'GEMFILE': 'ruby', 'RAKEFILE': 'ruby', 'BREWFILE': 'ruby',
            'PROCFILE': 'yaml', 'VAGRANTFILE': 'ruby',
            'CODEOWNERS': 'text'
        }
        return extensionless_languages.get(filename, "unknown")
    
    def scan_workspace(self) -> List[str]:
        """
        Scan workspace for code files.
        
        Returns:
            List of absolute file paths to code files
        """
        code_files = []
        
        for root, dirs, files in os.walk(self.workspace_path):
            # Filter out excluded directories in-place
            dirs[:] = [d for d in dirs if not self._should_exclude(os.path.join(root, d))]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if should be excluded
                if self._should_exclude(file_path):
                    continue
                
                # Check if it's a code file
                if not self._is_code_file(file_path):
                    continue
                
                # Check if binary
                if self._is_binary_file(file_path):
                    continue
                
                code_files.append(file_path)
        
        return code_files
    
    def chunk_file(self, file_path: str) -> List[CodeChunkIndexed]:
        """
        Parse and chunk a code file.
        
        Args:
            file_path: Absolute path to code file
            
        Returns:
            List of CodeChunkIndexed instances
        """
        chunks = []
        
        # Check file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return chunks
        
        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return chunks
        
        # Detect language
        language = self._detect_language(file_path)
        
        # Get relative path
        rel_path = os.path.relpath(file_path, self.workspace_path)
        rel_path = rel_path.replace("\\", "/")  # Normalize
        
        # Use AdaptiveChunker for intelligent file-type-aware chunking
        # This automatically selects the best strategy:
        # - Code files -> ASTChunker (semantic boundaries)
        # - Markdown/docs -> Section-based chunking (headers)
        # - Config files -> Structure-based chunking (keys/sections)
        # - Other files -> Paragraph/line-based chunking
        try:
            from ace.adaptive_chunker import AdaptiveChunker
            
            chunker = AdaptiveChunker()
            adaptive_chunks = chunker.chunk(content, file_path=file_path, language=language)
            
            for chunk in adaptive_chunks:
                chunks.append(CodeChunkIndexed(
                    content=chunk.content,
                    file_path=rel_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    language=chunk.language or language,
                    symbols=chunk.symbols if chunk.symbols else [],
                ))
        except ImportError:
            # Fallback to line-based chunking if adaptive_chunker not available
            lines = content.split("\n")
            chunk_size = 50
            overlap = 10
            
            for i in range(0, len(lines), chunk_size - overlap):
                chunk_lines = lines[i:i + chunk_size]
                if not chunk_lines:
                    continue
                
                chunks.append(CodeChunkIndexed(
                    content="\n".join(chunk_lines),
                    file_path=rel_path,
                    start_line=i + 1,
                    end_line=i + len(chunk_lines),
                    language=language,
                    symbols=[],
                ))
        except Exception as e:
            logger.warning(f"Failed to chunk {file_path}: {e}")
            # Fallback: single chunk for entire file
            lines = content.split("\n")
            chunks.append(CodeChunkIndexed(
                content=content,
                file_path=rel_path,
                start_line=1,
                end_line=len(lines),
                language=language,
                symbols=[],
            ))
        
        return chunks
    
    def index_file(self, file_path: str) -> int:
        """
        Index a single file into Qdrant with hybrid vectors.
        
        Args:
            file_path: Absolute path to file
            
        Returns:
            Number of chunks indexed
        """
        chunks = self.chunk_file(file_path)
        if not chunks:
            return 0
        
        # Get embedder
        embed_fn = self._get_embedder()
        
        # Import BM25 sparse vector function from unified_memory
        try:
            from ace.unified_memory import create_sparse_vector
        except ImportError:
            # Fallback: no sparse vectors
            create_sparse_vector = None
        
        # Generate points for Qdrant
        points = []
        try:
            from qdrant_client.models import PointStruct, SparseVector
            
            for chunk in chunks:
                dense_vector = embed_fn(chunk.content)
                
                # Build vector dict (named vectors for hybrid)
                vectors = {"dense": dense_vector}
                
                # Add sparse vector if available
                if create_sparse_vector:
                    sparse_data = create_sparse_vector(chunk.content)
                    if sparse_data["indices"]:
                        vectors["sparse"] = SparseVector(
                            indices=sparse_data["indices"],
                            values=sparse_data["values"],
                        )
                
                point = PointStruct(
                    id=abs(hash(chunk.chunk_hash)) % (2**63),  # Positive int64
                    vector=vectors,
                    payload={
                        "content": chunk.content,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "language": chunk.language,
                        "symbols": chunk.symbols,
                        "chunk_hash": chunk.chunk_hash,
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                points.append(point)
        except ImportError:
            logger.warning("qdrant-client not installed")
            return 0
        
        # Upsert to Qdrant
        if self._client and points:
            try:
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                
                # Track indexed file
                rel_path = os.path.relpath(file_path, self.workspace_path)
                self._indexed_files.add(rel_path)
                
            except Exception as e:
                logger.error(f"Failed to upsert to Qdrant: {e}")
                return 0
        
        return len(points)
    
    def index_workspace(self) -> Dict[str, Any]:
        """
        Index entire workspace with batch embedding for speed.
        
        Uses Voyage batch API to embed up to 128 chunks per request,
        making indexing ~100x faster than individual API calls.
        
        Returns:
            Statistics dict with files_indexed, chunks_indexed, etc.
        """
        stats = {
            "files_indexed": 0,
            "chunks_indexed": 0,
            "files_skipped": 0,
            "errors": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Scan for files
        code_files = self.scan_workspace()
        logger.info(f"Found {len(code_files)} code files to index")
        
        # Collect all chunks first
        all_chunks = []
        file_chunk_counts = {}  # Track chunks per file
        
        for file_path in code_files:
            try:
                chunks = self.chunk_file(file_path)
                if chunks:
                    start_idx = len(all_chunks)
                    all_chunks.extend(chunks)
                    file_chunk_counts[file_path] = (start_idx, len(chunks))
                    stats["files_indexed"] += 1
                else:
                    stats["files_skipped"] += 1
            except Exception as e:
                stats["errors"].append(f"{file_path}: {str(e)}")
                stats["files_skipped"] += 1
        
        if not all_chunks:
            stats["completed_at"] = datetime.now(timezone.utc).isoformat()
            return stats
        
        logger.info(f"Chunked {len(all_chunks)} total chunks from {stats['files_indexed']} files")
        
        # BATCH EMBED all chunks at once
        logger.info("Starting batch embedding...")
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self._embed_batch(texts)
        logger.info(f"Batch embedding complete: {len(embeddings)} vectors")
        
        # Import BM25 sparse vector function
        try:
            from ace.unified_memory import create_sparse_vector
        except ImportError:
            create_sparse_vector = None
        
        # Generate points for Qdrant
        try:
            from qdrant_client.models import PointStruct, SparseVector
        except ImportError:
            logger.warning("qdrant-client not installed")
            stats["completed_at"] = datetime.now(timezone.utc).isoformat()
            return stats
        
        points = []
        for chunk, dense_vector in zip(all_chunks, embeddings):
            vectors = {"dense": dense_vector}
            
            # Add sparse vector if available
            if create_sparse_vector:
                sparse_data = create_sparse_vector(chunk.content)
                if sparse_data["indices"]:
                    vectors["sparse"] = SparseVector(
                        indices=sparse_data["indices"],
                        values=sparse_data["values"],
                    )
            
            point = PointStruct(
                id=abs(hash(chunk.chunk_hash)) % (2**63),
                vector=vectors,
                payload={
                    "content": chunk.content,
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "symbols": chunk.symbols,
                    "chunk_hash": chunk.chunk_hash,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            points.append(point)
        
        # Batch upsert to Qdrant (in chunks of 100 for stability)
        if self._client and points:
            upsert_batch_size = 100
            for i in range(0, len(points), upsert_batch_size):
                batch = points[i:i + upsert_batch_size]
                try:
                    self._client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                    )
                except Exception as e:
                    logger.error(f"Failed to upsert batch to Qdrant: {e}")
                    stats["errors"].append(f"Qdrant upsert: {str(e)}")
            
            logger.info(f"Upserted {len(points)} points to Qdrant")
        
        stats["chunks_indexed"] = len(points)
        stats["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(f"Indexed {stats['files_indexed']} files, {stats['chunks_indexed']} chunks")
        
        return stats
    
    def update_file(self, file_path: str) -> int:
        """
        Update index for a single file (after modification).
        
        Args:
            file_path: Absolute path to file
            
        Returns:
            Number of chunks indexed
        """
        # Remove old chunks for this file first
        rel_path = os.path.relpath(file_path, self.workspace_path)
        rel_path = rel_path.replace("\\", "/")
        
        if self._client:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="file_path",
                                match=MatchValue(value=rel_path),
                            )
                        ]
                    ),
                )
            except Exception as e:
                logger.warning(f"Failed to remove old chunks: {e}")
        
        # Re-index file
        return self.index_file(file_path)
    
    def remove_file(self, rel_path: str) -> None:
        """
        Remove a file from the index.
        
        Args:
            rel_path: Relative path to file (from workspace root)
        """
        rel_path = rel_path.replace("\\", "/")
        
        if self._client:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="file_path",
                                match=MatchValue(value=rel_path),
                            )
                        ]
                    ),
                )
                
                self._indexed_files.discard(rel_path)
                
            except Exception as e:
                logger.warning(f"Failed to remove file from index: {e}")
    
    def get_indexed_files(self) -> List[str]:
        """
        Get list of indexed files.
        
        Returns:
            List of relative file paths
        """
        return list(self._indexed_files)
    
    def is_watching(self) -> bool:
        """Check if file watcher is active."""
        return self._watching
    
    def start_watching(self) -> None:
        """Start file watcher for auto-updates."""
        if self._watching:
            return
        
        self._watcher_stop_event.clear()
        self._watching = True
        
        # Try to use watchdog if available
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class CodeFileHandler(FileSystemEventHandler):
                def __init__(self, indexer):
                    self.indexer = indexer
                
                def on_modified(self, event):
                    if not event.is_directory and self.indexer._is_code_file(event.src_path):
                        self.indexer.update_file(event.src_path)
                
                def on_created(self, event):
                    if not event.is_directory and self.indexer._is_code_file(event.src_path):
                        self.indexer.index_file(event.src_path)
                
                def on_deleted(self, event):
                    if not event.is_directory:
                        rel_path = os.path.relpath(event.src_path, self.indexer.workspace_path)
                        self.indexer.remove_file(rel_path)
            
            self._observer = Observer()
            self._observer.schedule(
                CodeFileHandler(self),
                self.workspace_path,
                recursive=True,
            )
            self._observer.start()
            logger.info("Started file watcher")
            
        except ImportError:
            logger.warning("watchdog not installed, file watching disabled")
            self._watching = False
    
    def stop_watching(self) -> None:
        """Stop file watcher."""
        self._watcher_stop_event.set()
        self._watching = False
        
        if hasattr(self, "_observer"):
            try:
                self._observer.stop()
                self._observer.join(timeout=2)
            except Exception:
                pass
        
        logger.info("Stopped file watcher")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def index_workspace(workspace_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to index a workspace.
    
    Args:
        workspace_path: Root directory of workspace
        **kwargs: Additional arguments for CodeIndexer
        
    Returns:
        Indexing statistics
    """
    indexer = CodeIndexer(workspace_path=workspace_path, **kwargs)
    return indexer.index_workspace()
