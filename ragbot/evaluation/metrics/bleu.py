"""BLEU score metric for evaluating text similarity in RAG systems."""

from dataclasses import dataclass, field
from typing import Any, Set

from ragbot.evaluation.dataset_schema import Sample
from ragbot.evaluation.metrics.base import Metric


@dataclass
class BLEU(Metric):
    """BLEU score metric for comparing generated and reference answers.

    This implementation uses the `sacrebleu.corpus_bleu` method to compute
    a BLEU score for each answer-reference pair.

    Attributes:
        name (str): The name of the metric.
    """

    name: str = field(default="bleu", repr=True)
    _required_columns: Set[str] = field(
        default_factory=lambda: {"answer", "reference_answer"}
    )

    def __post_init__(self):
        """Initialize BLEU metric and ensure `sacrebleu` is available."""
        try:
            from sacrebleu import corpus_bleu
        except ImportError as e:
            raise ImportError(
                f"{e.name} is required. Please install it with `pip install {e.name}`"
            )
        self.corpus_bleu = corpus_bleu

    def score(self, sample: Sample, **kwargs: Any) -> float:
        """Compute BLEU score for a given sample.

        Args:
            sample (Sample): A sample containing `answer` and `reference_answer`.
            **kwargs: Optional keyword arguments (unused here).

        Returns:
            float: BLEU score as a float between 0 and 1.
        """
        reference, answer = sample.reference_answer, sample.answer
        ref_sentences = reference.split(". ")
        ans_sentences = answer.split(". ")

        reference = [[reference] for reference in ref_sentences]
        answer = ans_sentences
        score = self.corpus_bleu(answer, reference).score / 100
        return score
