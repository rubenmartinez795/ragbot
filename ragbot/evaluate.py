"""Evaluation pipeline for RAG model performance using LangSmith metrics."""

import json
import os
import time

import langsmith

from ragbot.evaluation.eval_chain import EvaluatorChain
from ragbot.evaluation.metrics.answer_relevance import AnswerRelevance
from ragbot.evaluation.metrics.bleu import BLEU
from ragbot.evaluation.metrics.context_relevance import ContextRelevance
from ragbot.evaluation.metrics.faithfulness import Faithfulness
from ragbot.evaluation.metrics.rouge import ROUGE
from ragbot.evaluation.metrics.semantic_similarity import SemanticSimilarity
from ragbot.rag import setup
from ragbot.utils.utils import get_embeddings, get_model


def evaluate(project_name: str, config_path: str, dataset_name: str):
    """Run evaluation on a RAG setup using LangSmith metrics.

    Loads a configuration file to initialize a RAG chain and evaluates its
    performance on a dataset using a set of standard metrics.

    Args:
        project_name: The name of the LangChain project used in LangSmith.
        config_path: Path to the JSON configuration file defining the RAG setup.
        dataset_name: The name of the dataset to be used for evaluation, registered in LangSmith.

    Raises:
        IOError: If the configuration file at `config_path` does not exist.
    """

    # Define LLM and embedding model for evalution
    llm = get_model("google", "gemini-2.0-flash", temperature=0.0)
    embeddings = get_embeddings("google", "models/embedding-001")

    # Wrap the metrics for evaluation with LangSmith
    evaluators = [
        EvaluatorChain(metric=metric, llm=llm, embeddings=embeddings)
        for metric in [
            BLEU(),
            ROUGE(),
            SemanticSimilarity(),
            Faithfulness(),
            AnswerRelevance(),
            ContextRelevance(),
        ]
    ]

    # Load configuration from json config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            metadata = config
    else:
        raise IOError(f"Configuration file {config_path} not found.")

    # Set up RAG, and run an evalution
    rag_chain = setup(
        project_name=project_name,
        llm_provider=config["llm_provider"],
        llm=config["llm"],
        llm_temperature=config["llm_temperature"],
        llm_top_p=config["llm_top_p"],
        llm_top_k=config["llm_top_k"],
        embeddings_provider=config["embeddings_provider"],
        embedding_model=config["embedding_model"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        search_type=config["search_type"],
        k_docs=config["k_docs"],
    )

    # Add delay for quota management
    def invoke_with_delay(*args, **kwargs):
        time.sleep(12)
        return rag_chain.invoke(*args, **kwargs)

    # Run evaluation
    langsmith.evaluate(
        invoke_with_delay,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="base",
        metadata=metadata,
        max_concurrency=1,
    )
