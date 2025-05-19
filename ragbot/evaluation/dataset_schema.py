"""Schema definitions for evaluation samples used in RAG evaluation."""

from typing import List, Optional

from pydantic import BaseModel


class BaseSample(BaseModel):
    """Base class for evaluation samples providing feature extraction."""

    def get_features(self):
        return list(self.to_dict().keys())


class Sample(BaseSample):
    """Structured sample used for evaluating RAG responses.

    Attributes:
        question (Optional[str]): The user's question or input.
        answer (Optional[str]): The model-generated answer.
        retrieved_context (Optional[List[str]]): The context retrieved by the RAG pipeline.
        reference_context (Optional[List[str]]): The ground truth context.
        reference_answer (Optional[str]): The ground truth answer.
    """

    question: Optional[str] = None
    answer: Optional[str] = None
    retrieved_context: Optional[List[str]] = None
    reference_context: Optional[List[str]] = None
    reference_answer: Optional[str] = None
