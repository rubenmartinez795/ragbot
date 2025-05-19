"""ROUGE score metric for evaluating similarity between generated and reference answers."""

from dataclasses import dataclass, field
from typing import Any, Literal, Set

from ragbot.evaluation.dataset_schema import Sample
from ragbot.evaluation.metrics.base import Metric


@dataclass
class ROUGE(Metric):
    """ROUGE score metric for RAG evaluation.

    Uses `rouge_score` to compute overlap-based metrics such as ROUGE-1 or ROUGE-L.
    You can configure the specific ROUGE type and the score mode (F1, precision, recall).

    Attributes:
        name (str): Name of the metric.
        rouge_type (str): Type of ROUGE to use ('rouge1' or 'rougeL').
        mode (str): Score mode ('fmeasure', 'precision', or 'recall').
    """

    name: str = field(default="rouge", repr=True)
    _required_columns: Set[str] = field(
        default_factory=lambda: {"answer", "reference_answer"}
    )
    rouge_type: Literal["rouge1", "rougeL"] = "rougeL"
    mode: Literal["fmeasure", "precision", "recall"] = "fmeasure"

    def __post_init__(self):
        """Initialize the ROUGE scorer and ensure required package is available."""
        try:
            from rouge_score import rouge_scorer
        except ImportError as e:
            raise ImportError(
                f"{e.name} is required. Please install it with `pip install {e.name}`"
            )
        self.rouge_scorer = rouge_scorer

    def score(self, sample: Sample, **kwargs: Any) -> float:
        """Compute ROUGE score for a given sample.

        Args:
            sample (Sample): Sample containing both `answer` and `reference_answer`.
            **kwargs: Optional keyword arguments (not used here).

        Returns:
            float: The ROUGE score (f-measure, precision, or recall depending on config).
        """
        scorer = self.rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)
        scores = scorer.score(sample.reference_answer, sample.answer)
        return getattr(scores[self.rouge_type], self.mode)
