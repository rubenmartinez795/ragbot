"""Interactive chat interface for a RAG-powered language model."""

from IPython.core.display import display_markdown

from ragbot.rag import setup


def chat(
    project_name: str,
    llm_provider: str,
    llm: str,
    llm_temperature: float,
    llm_top_p: float,
    llm_top_k: int,
    embeddings_provider: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    search_type: str,
    k_docs: int,
):
    """Start an interactive chat session using a RAG pipeline.

    This function initializes the RAG components with the given parameters and launches
    a loop that allows users to send queries to the model. It supports simple commands
    for help (`/?`), clearing history (`/clear`), and exiting (`/bye`).

    Args:
        project_name: The name of the LangChain project.
        llm_provider: The LLM provider (e.g., "google", "ollama", "hf").
        llm: The LLM model identifier.
        llm_temperature: Sampling temperature for the LLM.
        llm_top_p: Top-p nucleus sampling parameter.
        llm_top_k: Top-k sampling parameter.
        embeddings_provider: The provider for the embeddings model.
        embedding_model: The embeddings model identifier.
        chunk_size: Size of each document chunk for retrieval.
        chunk_overlap: Number of overlapping tokens between chunks.
        search_type: The type of retrieval search to use.
        k_docs: Number of top documents to retrieve for each query.
    """
    qa = setup(
        project_name=project_name,
        llm_provider=llm_provider,
        llm=llm,
        llm_temperature=llm_temperature,
        llm_top_p=llm_top_p,
        llm_top_k=llm_top_k,
        embeddings_provider=embeddings_provider,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        search_type=search_type,
        k_docs=k_docs,
    )

    history = []
    while True:
        query = input(">>> ")
        if query == "/?":
            print("Type /? to show this help")
            print("Type /bye to exit.")
            print("Type /clear to clear the chat history.")
            continue
        if query == "/clear":
            history = []
            continue
        if query.lower() == "/bye":
            print("Session closed!")
            exit(0)
        response = qa.invoke({"history": history, "input": query})
        history.extend([("human", query), ("ai", response["answer"])])
        display_markdown(response["answer"])
