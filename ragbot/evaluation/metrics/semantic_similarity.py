"""Semantic similarity metric for evaluating the similarity between generated and reference answers."""

from dataclasses import dataclass, field
from typing import Any, Set

import numpy as np

from ragbot.evaluation.dataset_schema import Sample
from ragbot.evaluation.metrics.base import MetricWithEmbeddings


@dataclass
class SemanticSimilarity(MetricWithEmbeddings):
    """Semantic similarity metric for RAG evaluation.

    This metric evaluates the semantic similarity between a generated answer and a reference answer
    by comparing their embeddings. The similarity is measured using cosine similarity between the
    two embedding vectors.

    Attributes:
        name (str): The name of the metric.
    """

    name: str = field(default="semantic similarity", repr=True)
    _required_columns: Set[str] = field(
        default_factory=lambda: {"answer", "reference_answer"}
    )

    def score(self, sample: Sample, **kwargs: Any) -> float:
        """Compute the semantic similarity score between the generated answer and the reference answer.

        This method evaluates the similarity between the answer and the reference answer by comparing
        their embeddings. It computes the cosine similarity between the two vectors.

        Args:
            sample (Sample): A sample containing the generated answer and the reference answer.
            **kwargs: Optional keyword arguments (not used here).

        Returns:
            float: A similarity score between 0 and 1 indicating the degree of similarity between the
                  generated answer and the reference answer.
        """
        reference, answer = sample.reference_answer, sample.answer

        # Handle embeddings for empty strings
        reference = reference or " "
        answer = answer or " "

        ref_embedding = np.array(self.embeddings.embed_query(reference))
        ans_embedding = np.array(self.embeddings.embed_query(answer))

        ref_norms = np.linalg.norm(ref_embedding, keepdims=True)
        ans_norms = np.linalg.norm(ans_embedding, keepdims=True)
        normalized_ref_emb = ref_embedding / ref_norms
        normalized_ans_emb = ans_embedding / ans_norms
        similarity = normalized_ref_emb @ normalized_ans_emb.T
        score = similarity.flatten()
        return score.tolist()[0]
