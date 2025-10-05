# Quick Start Guide

Welcome to the **RAGBOT** Quick Start Tutorial — a CLI tool for building and evaluating RAG-based (Retrieval-Augmented Generation) chatbots.

This guide walks you through:

- Project setup (including required dependencies and API keys)
- Connecting to third-party tools like LangSmith and Google AI Studio
- Running a hands-on example: building and evaluating a RAG chatbot

By the end, you’ll have a working example and understand how to adapt RAGBOT to your own use cases.


## 🛠️ Project Setup

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

### 4. Test Installation

Run the CLI to make sure everything is working:

```bash
poetry run python -m ragbot.cli -h
```

You should see the available CLI commands printed out.



## 🔐 API Setup

RAGBOT depends on two APIs: **LangSmith** (for logging and evaluation) and **Google AI Studio** (for model inference).

### ✅ LangSmith Setup

#### 1. Create a LangSmith Account

Go to [smith.langchain.com](https://smith.langchain.com/) and sign up.

#### 2. Generate an API Key

- Navigate to **Settings → API Keys**
- Click **New API Key**
- Choose **Personal Access Token** as the type

#### 3. Add the Key as an Environment Variable

On your machine, export the key:

```bash
export LANGCHAIN_API_KEY=<your_key_here>
```

> Or add it to your shell config (`.bashrc`, `.zshrc`, etc.)



### ✅ Google AI Studio API Setup

#### 1. Create a Google AI Key

Go to [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys) and generate a new API key.

#### 2. Add the Key as an Environment Variable

```bash
export GOOGLE_API_KEY=<your_key_here>
```



## 🚀 Hands-on Tutorial

Now that setup is complete, let’s run through a full example: building and evaluating a RAG chatbot for an application called **Laredo**.



### Step 1: Add Knowledge Base Files

Each project must have its own directory under `data/`, containing `.txt` files with relevant context.

For this example, we’ll use the pre-provided `example` project:

```
data/
└── example/
    ├── kb0.txt
    ├── ...
```

> 📁 Only `.txt` files are processed. Others will be ignored.



### Step 2: Define the System Prompt

Each project must include a `system.prompt` file with instructions for the chatbot. This prompt **must include** a `{context}` placeholder — it will be dynamically replaced by retrieved content.

For example: [`data/example/system.prompt`](data/example/system.prompt)

```text
You are the virtual assistant of the Laredo application. Your task is to answer users' questions regarding the use of
the application. To do so, please use the following context chunks retrieved from the Laredo application documentation.
If even with the provided context, you don't know the answer, simply say that you don't know.
Use at most three phrases to answer, and try to maintain your response concise.

{context}
```

> ⚠️ If `{context}` is missing, the RAG pipeline won't include retrieved documents.



### Step 3: Start a Chat Session

Launch an interactive session with your project:

```bash
poetry run python -m ragbot.cli chat -p example --temperature 1
```

Sample interaction:

```bash
>>> Hello!
Hi! How can I help you with the Laredo application?

>>> What file formats does Laredo accept for the dataset?
The application accepts files in CSV format.
```

To end the session:

```bash
>>> /bye
Session closed!
```

You can now view the execution trace in your [LangSmith dashboard](https://smith.langchain.com/) under **Tracing Projects → example**.



### Step 4: Evaluate with a Dataset

Let’s evaluate the chatbot’s performance by comparing its responses to a set of reference answers.

#### 1. Build the Evaluation Dataset

Use the helper script to create a LangSmith dataset (`ds-example`) with four questions and reference answers:

```bash
poetry run python -m scripts.build_dataset
```

This only needs to be done **once**. You can edit the dataset later from LangSmith's **Datasets** panel.



#### 2. Create or Modify a Configuration File

Experiments use configuration files to specify model parameters and RAG settings.

For this tutorial, use the provided one:

```
configs/
└── default.json
```



#### 3. Run the Experiment

Use the following command:

```bash
poetry run python -m ragbot.cli evaluate -p example --dataset-name ds-example --config-path configs/default.json
```

RAGBOT will:

- Retrieve answers using your RAG setup
- Compare them to the reference answers
- Log the results to LangSmith with detailed metrics (e.g., latency, tokens, cost, evaluation scores)

You can now review results in LangSmith under:

> **Datasets & Experiments → ds-example → [Your Experiment]**



## 🔁 Iterating

You can repeat experiments with different configs to compare setups and optimize performance.

Just create new config files under `configs/` and run:

```bash
poetry run python -m ragbot.cli evaluate -p example --dataset-name ds-example --config-path configs/<your_config>.json
```



## ✅ You’re All Set!

You now know how to:

- Set up RAGBOT with LangSmith and Google AI
- Load custom data and prompts
- Run a chat session with your RAG pipeline
- Evaluate it using automated experiments

To go further, check out:

- [README.md](README.md) — High-level overview
- [DEVELOPMENT.md](DEVELOPMENT.md) — For contributors
