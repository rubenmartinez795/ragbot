# RAGBOT CLI
```
@author Rubén Martínez Amodia
@email ruben.martinezam@alumnos.unican.es
```

RAGBOT is a CLI-based framework developed as part of a bachelor thesis project. It supports
the experimentation and evaluation of Retrieval-Augmented Generation (RAG) systems,
allowing for flexible configuration, execution, and analysis of chatbot behavior.

The tool is built on top of the LangChain framework and integrates with LangSmith for
experiment tracking and logging. Its primary goal is to facilitate the reproducible
development and evaluation of RAG-based virtual assistants.

## Features

- Interactive testing of RAG chatbot pipelines
- Configuration of large language models (LLMs), embedding models, retrievers, and other components
- Modular evaluation framework with configurable metrics
- Logging and observability through LangSmith integration

## Evaluation Capabilities

RAGBOT provides a structured approach to evaluation through:

- LLM-based scoring (e.g., LLM-as-a-judge)
- Embedding-based similarity metrics
- Additional customizable evaluation components for domain-specific needs

These tools are intended to support both qualitative and quantitative assessment
of chatbot performance.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rubenmartinez795/ragbot.git
cd ragbot
```

### 2. Install Dependencies

RAGBOT uses [Poetry](https://python-poetry.org/) for dependency management. Firstly, install
poetry.
```bash
pip install poetry
```

Then, install the project dependencies running:
```bash
poetry install
```
This will automatically create a virtual environment in your machine. To see where it is, do:
```bash
poetry env info --path
```

### 3. Activate the Virtual Environment

```bash
poetry env activate
```

## Usage

RAGBOT provides a command-line interface to interact with and evaluate RAG chatbots.
The CLI supports two primary commands:

1. **`chat`** – Launch an interactive chatbot session with configurable RAG parameters.  
2. **`evaluate`** – Perform automated evaluation of RAG configurations using datasets and metrics.

To run any CLI command, use:

```bash
poetry run python -m ragbot.cli <command> [options]
```

You can append `-h` or `--help` to any command to display help information.

### `chat`

Starts an interactive RAG chatbot session for manual testing and experimentation.

```bash
poetry run python -m ragbot.cli chat -p <project> [options]
```

#### Required argument

- `-p, --proj`: Name of the project to run (e.g., `flotty`, `h2o`)

#### Optional arguments

- `--llm-provider`: Name of the LLM provider (e.g., `google`)
- `--llm`: Identifier of the LLM model (e.g., `gemini-2.0-flash`)
- `--temperature`: Sampling temperature
- `--top-p`: Nucleus sampling value
- `--top-k`: Number of top tokens considered
- `--emb-provider`: Embeddings provider (e.g., `google`)
- `--emb-model`: Embedding model identifier (e.g., `models/embedding-001`)
- `--chunk-size`: Chunk size for the text splitter
- `--chunk-overlap`: Overlap size between text chunks
- `--search-type`: Retrieval search strategy
- `--k`: Number of documents to retrieve

This command enables interactive chatbot testing while allowing control over core RAG parameters.

### `evaluate`

Performs evaluation of a RAG chatbot setup based on a given dataset and configuration.

```bash
poetry run python -m ragbot.cli evaluate -p <project> [options]
```

#### Required argument

- `-p, --proj`: Name of the project to evaluate

#### Optional arguments

- `--config-path`: Path to a JSON file specifying the evaluation configuration
- `--dataset-name`: Name of the LangSmith dataset to use

This command supports automated performance analysis with various metrics such as BLEU, ROUGE, context relevance, and faithfulness.

### Help

To get help on any command, use:

```bash
poetry run python -m ragbot.cli <command> --help
```

Example:

```bash
poetry run python -m ragbot.cli chat --help
```

## Documentation
Full documentation is available at [RAGBOT Doc](https://google.com).

## Development
For development guidelines and instructions on how to contribute, see [DEVELOPMENT.md](DEVELOPMENT.md).
