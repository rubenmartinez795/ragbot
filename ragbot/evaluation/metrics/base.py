"""Base classes for defining RAG evaluation metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Set

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM, BaseChatModel

from ragbot.evaluation.dataset_schema import Sample


@dataclass
class Metric(ABC):
    """Abstract base class for evaluation metrics.

    Attributes:
        name (str): Name of the metric.
    """

    name: str = field(default="", repr=True)
    _required_columns: Set[str] = field(default_factory=set)

    def init(self):
        """Optional initializer hook for the metric."""
        pass

    @property
    def required_columns(self) -> Set[str]:
        """Set of columns required in the input sample to compute the metric.

        Returns:
            Set[str]: Required column names.
        """
        return self._required_columns

    @required_columns.setter
    def required_columns(self, required_columns: Set[str]):
        """Set the required columns.

        Args:
            required_columns (Set[str]): Column names to be required.
        """
        self._required_columns = required_columns

    @abstractmethod
    def score(self, sample: Sample, **kwargs: Any) -> float:
        """Compute a score for a given sample.

        Args:
            sample (Sample): Input data to evaluate.
            **kwargs: Additional arguments (e.g., for callbacks or config).

        Returns:
            float: Computed metric score.
        """
        pass


class MetricWithLLM(Metric, ABC):
    """Base class for metrics that require an LLM to operate."""

    llm: BaseChatModel | LLM = None

    def init(self):
        """Check that the LLM is set before scoring."""
        if self.llm is None:
            raise ValueError(f"LLM not set for Metric {self.name}")


class MetricWithEmbeddings(Metric, ABC):
    """Base class for metrics that require embeddings to operate."""

    embeddings: Embeddings = None

    def init(self):
        """Check that embeddings are set before scoring."""
        if self.embeddings is None:
            raise ValueError(f"Embeddings not set for Metric {self.name}")
