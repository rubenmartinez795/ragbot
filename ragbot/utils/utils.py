"""Utilities for configuring and loading language models and embeddings."""

import os

import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM, BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_ollama import ChatOllama


def use_langsmith(project_name: str):
    """Configure LangSmith for experiment tracking.

    Sets environment variables to enable LangSmith's advanced tracing and logging features
    for the given project. Note that you must have the environment variable LANGCHAIN_API_KEY
    set to use this.

    Args:
        project_name: Name of the LangSmith project to associate traces with.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = project_name


def get_model(
    provider: str,
    model_id: str,
    temperature: float,
    top_p: float = 0.85,
    top_k: int = 40,
) -> BaseChatModel | LLM:
    """Load a chat language model from the specified provider.

    Supports models from Google Generative AI, HuggingFace, and Ollama. Note that for Google
    Generative AI you will need to have the GOOGLE_API_KEY environment variable set, and for
    HuggingFace you will need the HUGGINGFACEHUB_API_TOKEN environment variable set.

    Args:
        provider: Name of the LLM provider. Supported values: "google", "ollama", "hf".
        model_id: Model identifier or repository ID (e.g., "gemini-1.5-flash").
        temperature: Sampling temperature to use (0.0 for deterministic output).
        top_p: Nucleus sampling threshold. Only applicable to some providers.
        top_k: Number of top tokens to consider. Only applicable to some providers.

    Returns:
        An instance of a language model (LLM or BaseChatModel).

    Raises:
        ValueError: If the provider is not recognized.
    """
    if provider == "google":
        # Configure Google AI Studio API key
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        llm = ChatGoogleGenerativeAI(
            model=model_id, temperature=temperature, top_p=top_p, top_k=top_k
        )
    elif provider == "ollama":
        llm = ChatOllama(model=model_id, temperature=temperature)
    elif provider == "hf":
        llm = HuggingFaceEndpoint(repo_id=model_id, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return llm


def get_embeddings(provider: str, model_id: str) -> Embeddings:
    """Load an embeddings model from the specified provider.

    Currently only supports Google Generative AI embeddings.

    Args:
        provider: Name of the embedding provider. Supported value: "google".
        model_id: Identifier of the embedding model (e.g., "models/embedding-001").

    Returns:
        An instance of a LangChain-compatible embeddings model.

    Raises:
        ValueError: If the provider is not recognized.
    """
    if provider == "google":
        embeddings = GoogleGenerativeAIEmbeddings(model=model_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return embeddings
