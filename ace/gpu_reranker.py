"""GPU-Accelerated Cross-Encoder Reranker using ONNX Runtime + DirectML.

Supports AMD GPUs via DirectML on Windows.
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for ONNX Runtime with DirectML
try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
    DML_AVAILABLE = 'DmlExecutionProvider' in ort.get_available_providers()
except ImportError:
    ONNX_AVAILABLE = False
    DML_AVAILABLE = False

# Fallback to sentence-transformers
try:
    from sentence_transformers import CrossEncoder
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class GPUCrossEncoder:
    """Cross-encoder with GPU acceleration via ONNX Runtime + DirectML."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_gpu: bool = True,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.use_gpu = use_gpu and DML_AVAILABLE
        self._session = None
        self._tokenizer = None
        self._fallback_model = None

        if self.use_gpu and ONNX_AVAILABLE:
            self._init_onnx()
        elif ST_AVAILABLE:
            self._init_fallback()
        else:
            raise ImportError("No cross-encoder backend available")

    def _init_onnx(self):
        """Initialize ONNX Runtime with DirectML."""
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification

            logger.info(f"Loading ONNX model with DirectML: {self.model_name}")

            # Export or load ONNX model
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Try to load optimized ONNX model, fall back to auto-export
            try:
                self._ort_model = ORTModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    export=True,
                    provider="DmlExecutionProvider"
                )
                logger.info("ONNX model loaded with DirectML GPU acceleration")
                return
            except Exception as e:
                logger.warning(f"ONNX with DirectML failed: {e}, trying CPU fallback")

        except ImportError:
            logger.warning("optimum not available, falling back to sentence-transformers")

        # Fall back to sentence-transformers
        self._init_fallback()

    def _init_fallback(self):
        """Initialize fallback sentence-transformers model."""
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers not available")

        logger.info(f"Loading CrossEncoder (CPU): {self.model_name}")
        self._fallback_model = CrossEncoder(self.model_name, max_length=self.max_length)
        self.use_gpu = False
        logger.info("CrossEncoder loaded (CPU mode)")

    def predict(self, sentence_pairs: List[List[str]]) -> List[float]:
        """
        Predict relevance scores for query-document pairs.

        Args:
            sentence_pairs: List of [query, document] pairs

        Returns:
            List of relevance scores
        """
        if not sentence_pairs:
            return []

        if self._fallback_model is not None:
            return self._fallback_model.predict(sentence_pairs).tolist()

        if hasattr(self, '_ort_model') and self._ort_model is not None:
            # Use ONNX model
            scores = []
            for query, doc in sentence_pairs:
                inputs = self._tokenizer(
                    query, doc,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="np"
                )
                outputs = self._ort_model(**inputs)
                score = outputs.logits[0][0].item() if hasattr(outputs.logits, 'item') else float(outputs.logits[0][0])
                scores.append(score)
            return scores

        raise RuntimeError("No model initialized")


def create_reranker(use_gpu: bool = True) -> GPUCrossEncoder:
    """Create a cross-encoder reranker with optional GPU acceleration."""
    return GPUCrossEncoder(use_gpu=use_gpu)


# Test GPU availability
def test_gpu():
    """Test if GPU acceleration is available."""
    print(f"ONNX Runtime available: {ONNX_AVAILABLE}")
    print(f"DirectML available: {DML_AVAILABLE}")
    print(f"Sentence-Transformers available: {ST_AVAILABLE}")

    if DML_AVAILABLE:
        providers = ort.get_available_providers()
        print(f"ONNX providers: {providers}")


if __name__ == "__main__":
    test_gpu()
