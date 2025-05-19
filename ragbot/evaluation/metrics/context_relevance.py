"""Context relevance metric for evaluating the relevance of retrieved context."""

from dataclasses import dataclass, field
from typing import Any, Set

from ragbot.evaluation.dataset_schema import Sample
from ragbot.evaluation.metrics.base import MetricWithLLM


@dataclass
class ContextRelevance(MetricWithLLM):
    """Context relevance metric for RAG evaluation.

    This metric evaluates the relevance of the retrieved context to the user's question
    based on a scoring system from 1 to 5. The metric uses an LLM to assess how well
    the context aligns with the question.

    Attributes:
        name (str): The name of the metric.
    """

    name: str = field(default="context relevance", repr=True)
    _required_columns: Set[str] = field(
        default_factory=lambda: {"question", "retrieved_context"}
    )

    def score(self, sample: Sample, **kwargs: Any) -> float:
        """Compute the context relevance score for a given sample.

        Args:
            sample (Sample): A sample containing the user question and retrieved context.
            **kwargs: Optional keyword arguments (not used here).

        Returns:
            float: A score (1-5) indicating how relevant the retrieved context is to the user's question.
        """
        question, context = sample.question, sample.retrieved_context
        output = self.llm.invoke(
            f"""
            You are an expert evaluator assessing how relevant a retrieved context is to a given user question. 
            Context relevance is defined as how well the retrieved information aligns with the question. 
        
            **Scoring Guidelines:**  
            - **5 (Excellent):** The context is fully relevant to the question, containing directly useful and necessary information.  
            - **4 (Good):** The context is mostly relevant but may include minor irrelevant details or miss slight nuances.  
            - **3 (Acceptable):** The context is somewhat relevant but may include noticeable irrelevant parts or lack some important details.  
            - **2 (Poor):** The context is only partially relevant, with significant irrelevant or missing information.  
            - **1 (Not Relevant):** The context is mostly or entirely unrelated to the question.  
        
            **User Question:** {question}  
            **Retrieved Context:** {context}  
        
            Assign a single integer score (1-5) based on the above criteria.
            Only return the score as a number, without any extra text.
    
            Verdict [1 | 2 | 3 | 4 | 5]: 
            """
        )

        score = int(output.content.strip())
        return score
