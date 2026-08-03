"""Task-specific observed-data adapters."""

from .network_interference import adapt_interference
from .observed_regime import adapt_regime
from .reference_benchmarks import ReferenceBenchmarkResult, load_reference_result
from .static_ate import adapt_ate

__all__ = [
    "ReferenceBenchmarkResult",
    "adapt_ate",
    "adapt_interference",
    "adapt_regime",
    "load_reference_result",
]
