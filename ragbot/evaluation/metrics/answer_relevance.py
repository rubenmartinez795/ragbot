"""Evaluation metric for answer relevance in RAG systems."""

from dataclasses import dataclass, field
from typing import Any, Set

from ragbot.evaluation.dataset_schema import Sample
from ragbot.evaluation.metrics.base import MetricWithLLM


@dataclass
class AnswerRelevance(MetricWithLLM):
    """Metric to evaluate the relevance of a generated answer to a user question.

    Attributes:
        name (str): The name of the metric.
    """

    name: str = field(default="answer relevance", repr=True)
    _required_columns: Set[str] = field(default_factory=lambda: {"question", "answer"})

    def score(self, sample: Sample, **kwargs: Any) -> float:
        """Score the answer relevance using a language model.

        The score is based on a scale from 1 to 5:
            5 - Excellent
            4 - Good
            3 - Acceptable
            2 - Poor
            1 - Not Relevant

        Args:
            sample (Sample): A sample containing a question and generated answer.
            **kwargs: Additional keyword arguments (e.g., for callbacks).

        Returns:
            float: A relevance score between 1 and 5.
        """
        question, answer = sample.question, sample.answer
        output = self.llm.invoke(
            f"""
            You are an expert evaluator assessing how relevant a given answer is to a user question. 
            Answer relevance is defined as how well the response aligns with the user input. 
        
            **Scoring Guidelines:**  
            - **5 (Excellent):** The answer fully addresses the question with clear, relevant, and complete information.  
            - **4 (Good):** The answer is mostly relevant but may miss minor details or include slight redundancy.  
            - **3 (Acceptable):** The answer is somewhat relevant but may be incomplete or contain noticeable redundancy.  
            - **2 (Poor):** The answer only partially addresses the question and includes significant irrelevant or missing content.  
            - **1 (Not Relevant):** The answer does not address the question or is entirely off-topic.  
        
            **User Question:** {question}  
            **Generated Answer:** {answer}  

            Assign a single integer score (1-5) based on the above criteria.
            Only return the score as a number, without any extra text.
            
            Verdict [1 | 2 | 3 | 4 | 5]: 
            """
        )

        score = int(output.content.strip())
        return score
