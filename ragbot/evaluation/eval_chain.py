"""EvaluatorChain for assessing the performance of a RAG pipeline."""

from typing import Any, Dict, List, Optional, Union, cast

from langchain.chains.base import Chain
from langchain.schema import RUN_KEY
from langchain_core.callbacks import CallbackManagerForChainRun
from langsmith import EvaluationResult, RunEvaluator
from langsmith.evaluation import EvaluationResults
from langsmith.schemas import Example, Run

from ragbot.evaluation.dataset_schema import Sample
from ragbot.evaluation.metrics.base import Metric, MetricWithEmbeddings, MetricWithLLM


class EvaluatorChain(Chain, RunEvaluator):
    """Chain used to evaluate RAG outputs using custom metrics.

    This chain wraps around a scoring metric (e.g., semantic similarity, BLEU)
    and applies it to a structured sample. It is compatible with LangSmith's
    evaluation framework.

    Attributes:
        metric (Metric): The scoring metric used for evaluation. Can support LLM or embedding models.
    """

    metric: Metric

    def __init__(self, metric: Metric, **kwargs: Any):
        """Initializes the EvaluatorChain.

        Args:
            metric (Metric): The evaluation metric instance.
            **kwargs: Optional keyword arguments, including 'llm' or 'embeddings' if required by the metric.
        """
        kwargs["metric"] = metric
        super().__init__(**kwargs)
        if isinstance(self.metric, MetricWithLLM):
            cast(MetricWithLLM, self.metric).llm = kwargs.get("llm")
        if isinstance(self.metric, MetricWithEmbeddings):
            cast(MetricWithEmbeddings, self.metric).embeddings = kwargs.get(
                "embeddings"
            )
        self.metric.init()

    @property
    def input_keys(self) -> List[str]:
        """Defines input keys required by the metric.

        Returns:
            List[str]: Required column names for the metric.
        """
        return list(self.metric.required_columns)

    @property
    def output_keys(self) -> List[str]:
        """Defines the output keys produced by the evaluation.

        Returns:
            List[str]: The name of the metric used as the output key.
        """
        return [self.metric.name]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Runs the evaluation chain with provided inputs.

        Args:
            inputs (Dict[str, Any]): Input dictionary for the metric.
            run_manager (Optional[CallbackManagerForChainRun]): Callback manager for tracking.

        Returns:
            Dict[str, Any]: The computed evaluation score.
        """
        sample = Sample(**inputs)
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        score = self.metric.score(sample, callbacks=callbacks)
        print(f"{self.metric.name}: {score}")
        return {self.metric.name: score}

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> Union[EvaluationResult, EvaluationResults]:
        """Evaluates a single run against a reference example.

        Args:
            run (Run): The RAG pipeline run to evaluate.
            example (Optional[Example]): Ground truth data for evaluation.

        Returns:
            Union[EvaluationResult, EvaluationResults]: Evaluation results for the run.
        """
        chain_eval = self._chain_eval(example, run)
        eval_output = self.invoke(chain_eval, include_run_info=True)
        eval_result = EvaluationResult(
            key=self.metric.name,
            score=eval_output[self.metric.name],
        )

        # Make the view of the run available in LangSmith
        if RUN_KEY in eval_output:
            eval_result.evaluator_info[RUN_KEY] = eval_output[RUN_KEY]

        return eval_result

    def _chain_eval(self, example: Example, run: Run) -> Dict:
        """Builds input dictionary for evaluation from run and example.

        Args:
            example (Example): Ground truth sample.
            run (Run): Output run to be evaluated.

        Returns:
            Dict: Input values expected by the metric.
        """
        chain_eval = dict()

        if "question" in self.metric.required_columns:
            chain_eval["question"] = run.outputs["input"]
        if "retrieved_context" in self.metric.required_columns:
            chain_eval["retrieved_context"] = [
                context.page_content for context in run.outputs["context"]
            ]
        if "reference_context" in self.metric.required_columns:
            chain_eval["reference_context"] = example.outputs["context"]
        if "answer" in self.metric.required_columns:
            chain_eval["answer"] = run.outputs["answer"]
        if "reference_answer" in self.metric.required_columns:
            chain_eval["reference_answer"] = example.outputs["output"]

        return chain_eval
