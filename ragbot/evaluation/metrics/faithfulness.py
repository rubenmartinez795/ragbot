"""Faithfulness metric for evaluating the accuracy of generated answers based on the retrieved context."""

from dataclasses import dataclass, field
from typing import Any, Set

from ragbot.evaluation.dataset_schema import Sample
from ragbot.evaluation.metrics.base import MetricWithLLM


@dataclass
class Faithfulness(MetricWithLLM):
    """Faithfulness metric for RAG evaluation.

    This metric evaluates how well the generated answer is grounded in the provided retrieved context.
    It uses an LLM to score the answer based on the degree to which it reflects the information in the context.

    Attributes:
        name (str): The name of the metric.
    """

    name: str = field(default="faithfulness", repr=True)
    _required_columns: Set[str] = field(
        default_factory=lambda: {"answer", "retrieved_context"}
    )

    def score(self, sample: Sample, **kwargs: Any) -> float:
        """Compute the faithfulness score for a given sample.

        This method evaluates how grounded the generated answer is in the provided context.

        Args:
            sample (Sample): A sample containing the retrieved context and generated answer.
            **kwargs: Optional keyword arguments (not used here).

        Returns:
            float: A score (1-5) indicating how well the answer is grounded in the retrieved context.
        """

        context, answer = sample.retrieved_context, sample.answer
        output = self.llm.invoke(
            f"""
            You are an expert evaluator assessing how well a generated answer is grounded in the provided retrieved context. 
            Groundedness is defined as how accurately the answer reflects the information in the retrieved context, without adding unsupported details or hallucinating information.
        
            **Scoring Guidelines:**  
            - **5 (Excellent):** The answer is fully grounded in the context, with no hallucinations or unsupported claims.  
            - **4 (Good):** The answer is mostly grounded but may have minor phrasing variations or slight extrapolations.  
            - **3 (Acceptable):** The answer is somewhat grounded but includes noticeable extrapolations or minor unsupported details.  
            - **2 (Poor):** The answer contains significant ungrounded information or misinterprets key details from the context.  
            - **1 (Not Grounded):** The answer is mostly or entirely hallucinated, containing little to no support from the provided context.  
        
            **Retrieved Context:** {context}  
            **Generated Answer:** {answer}  
        
            Assign a single integer score (1-5) based on the above criteria.
            Only return the score as a number, without any extra text.

            Verdict [1 | 2 | 3 | 4 | 5]: 
            """
        )

        score = int(output.content.strip())
        return score
