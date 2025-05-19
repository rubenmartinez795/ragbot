# CLI Overview

RAGBOT provides a command-line interface to interact with and evaluate Retrieval-Augmented Generation (RAG) chatbots. The CLI supports two main functionalities:

1. **Chat** with a RAG-powered virtual assistant using configurable parameters.
2. **Evaluate** the performance of different configurations or models using a dataset and metrics.

---

## Basic Usage

To run any CLI command, use:

```bash
poetry run python -m ragbot.cli <command> [options]
```

You can display help information by appending `-h` or `--help` to any command.

---

## Available Commands

### `chat`

Launches an interactive RAG chatbot session.

#### Usage

```bash
poetry run python -m ragbot.cli chat -p <project> [options]
```

#### Required argument

- `-p, --proj`: The name of the project to run.

#### Optional arguments

- `--llm-provider`: LLM provider name (e.g., `google`).
- `--llm`: LLM model name (e.g., `gemini-2.0-flash`).
- `--temperature`: Temperature setting for sampling.
- `--top-p`: Nucleus sampling value.
- `--top-k`: Number of top tokens to consider.
- `--emb-provider`: Embeddings provider (e.g., `google`).
- `--emb-model`: Embedding model name (e.g., `models/embedding-001`).
- `--chunk-size`: Chunk size for the text splitter.
- `--chunk-overlap`: Overlap between chunks.
- `--search-type`: Type of retrieval search.
- `--k`: Number of documents to retrieve.

This command lets you test chatbot behavior interactively while tweaking RAG parameters.

---

### `evaluate`

Evaluates a RAG chatbot setup using a given dataset and configuration.

#### Usage

```bash
poetry run python -m ragbot.cli evaluate -p <project> [options]
```

#### Required argument

- `-p, --proj`: The name of the project to evaluate.

#### Optional arguments

- `--config-path`: Path to a JSON configuration file defining the evaluation setup.
- `--dataset-name`: Name of the LangSmith dataset to evaluate on.

This command performs automated evaluations using metrics like BLEU, ROUGE, context relevance, faithfulness, and more.

---

## Help and Subcommands

For help on a specific command, run:

```bash
poetry run python -m ragbot.cli <command> --help
```

For example:

```bash
poetry run python -m ragbot.cli chat --help
```

This will display detailed options and usage instructions for that subcommand.

---

Refer to the [API Reference](reference/evaluate.md) for more information on available configuration options and evaluation metrics.
