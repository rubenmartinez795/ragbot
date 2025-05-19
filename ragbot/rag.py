"""RAG setup module for initializing retrieval-augmented generation chains."""

import glob
import os
import shutil

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbot.utils.utils import get_embeddings, get_model


def setup(
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
) -> Runnable:
    """Set up and return a RAG retrieval chain.

    Initializes a language model and embeddings, chunks input documents,
    and creates a retrieval-augmented generation chain using LangChain.

    Args:
        project_name: Name of the project..
        llm_provider: Provider name for the language model (e.g., "google", "ollama").
        llm: Identifier or model name for the language model.
        llm_temperature: Sampling temperature for language generation.
        llm_top_p: Nucleus sampling top-p value.
        llm_top_k: Top-k sampling value.
        embeddings_provider: Provider name for embeddings.
        embedding_model: Identifier for the embeddings model.
        chunk_size: Maximum number of characters per document chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        search_type: Type of search for the retriever (e.g., "similarity", "mmr").
        k_docs: Number of top documents to retrieve.

    Returns:
        A `Runnable` LangChain object that processes user input through a RAG pipeline.

    Raises:
        FileNotFoundError: If there is no `system.prompt` file at `data/project_name`.
        RuntimeError: If the system prompt file cannot be read, or if it does not contain the required `{context}` placeholder.
    """
    # Set up a language model
    llm = get_model(
        llm_provider, llm, temperature=llm_temperature, top_p=llm_top_p, top_k=llm_top_k
    )

    # Create documents from knowledge base
    docs = create_docs(project_name, chunk_size, chunk_overlap)

    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(embeddings_provider, embedding_model),
    )

    # Instantiate the relevant docs retriever
    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs={"k": k_docs}
    )

    # Check system prompt and read
    file = f"data/{project_name}/system.prompt"
    if not os.path.exists(file):
        raise FileNotFoundError(
            f"No system prompt for project {project_name}, please add file {file}"
        )

    try:
        with open(file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read file {file}: {e}") from e

    if not "{context}" in system_prompt:
        raise RuntimeError(
            f"System prompt at {file} must have the {{context}} placeholder"
        )

    # Create custom prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(
                variable_name="history",
                optional=True,
                n_messages=10,
            ),
            ("human", "{input}"),
        ]
    )

    # Create retrieval chain
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain


def create_docs(
    project_name: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Create and return documents for a given project.

    Loads, splits, and stores the content to be used for embedding and
    retrieval. The `project_name/` directory must exist under the `data`
    directory with its knowledge base of .txt files.

    Args:
        project_name: The project identifier.
        chunk_size: The maximum size of each text chunk.
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of `Document` objects.

    Raises:
        ValueError: If the `project_name` knowledge base is not found at data/.
        FileNotFoundError: If the `project_name` knowledge base is empty.
        RuntimeError: If it fails to read any file from the knowledge base.
    """
    dir_path = f"data/{project_name}"
    if not os.path.exists(dir_path):
        raise ValueError(
            f"No data for project {project_name}, please add directory {dir_path} with knowledge base files"
        )

    # Remove previous content
    content_dir = f"{dir_path}/content/"
    if os.path.exists(content_dir):
        shutil.rmtree(content_dir)
    os.makedirs(content_dir)

    # Declare splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Create documents
    docs = []
    knowledge_base = glob.glob(f"{dir_path}/*.txt")
    if not knowledge_base:
        raise FileNotFoundError(f"No .txt files found in {dir_path}")

    for file in knowledge_base:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file {file}: {e}") from e
        splits = text_splitter.split_text(content)
        docs.extend([Document(page_content=chunk) for chunk in splits])

    # Store documents
    for i, doc in enumerate(docs):
        with open(f"{dir_path}/content/doc{i}.txt", "w", encoding="utf-8") as f:
            f.write(doc.page_content)

    return docs
