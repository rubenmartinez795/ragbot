# Development Guide

Welcome to the development guide for **RAGBOT** — a CLI tool for building and evaluating
RAG-based chatbots. This document outlines how to set up the project locally, the
development workflow, testing practices, and contribution guidelines.

---

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

### 4. Check Run

```bash
poetry run python -m ragbot.cli -h
```

---

## 📁 Project Structure

```bash
ragbot/
│
├── ragbot/              # Core package
│   ├── __init__.py
│   ├── chat.py              # Chat interface logic
│   ├── cli.py               # CLI entry point and command parsing
│   ├── evaluate.py          # Evaluation execution entry
│   ├── rag.py               # Main RAG pipeline
│   ├── evaluation/          # Evaluation components
│   │   ├── __init__.py
│   │   ├── dataset_schema.py      # Dataset schema for evaluation
│   │   ├── eval_chain.py          # LLM evaluation chain
│   │   └── metrics/               # Built-in metrics
│   │       ├── __init__.py
│   │       ├── answer_relevance.py
│   │       ├── base.py
│   │       ├── bleu.py
│   │       ├── context_relevance.py
│   │       ├── faithfulness.py
│   │       ├── rouge.py
│   │       └── semantic_similarity.py
│   └── utils/
│       ├── __init__.py
│       └── utils.py          # General utility functions
│
├── tests/                # Unit tests
│
├── docs/                 # MkDocs documentation
│
├── configs/              # Evaluation config files
│
├── scripts/              # Some useful scripts
│
├── poetry.lock           # Project dependencies lock file (read-only)
├── pyproject.toml        # Project dependencies and configuration
├── README.md             # Project overview
├── DEVELOPMENT.md        # Developer setup and workflow
└── mkdocs.yml            # Documentation configuration
```

---

## 🚀 Development Workflow

1. **Branch from `main`**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your changes**, following best practices and consistent coding style.

3. **Run tests**:
   ```bash
   pytest
   ```

4. **Format code**:
   ```bash
   black .
   isort .
   ```

5. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: add your feature"
   git push origin feature/your-feature-name
   ```

---

## 🧪 Testing

- Tests are located under the `tests/` directory.
- Use `pytest` to run tests:
  ```bash
  poetry run pytest
  ```
- Include test cases for any new functionality.

---

## 📚 Documentation

- Documentation lives in the `docs/` folder and is built using [MkDocs](https://www.mkdocs.org/).
- To serve locally:
  ```bash
  poetry run mkdocs serve
  ```

---

## 🤝 Contributing

Even though this is part of an academic project, feel free to suggest improvements or open issues.

When contributing:

- Follow clear commit messages (e.g., `fix:`, `feat:`, `refactor:`).
- Write unit tests for new code.
- Ensure documentation is updated if applicable.

---

## 📬 Contact

Developed by **Rubén Martínez Amodia**  
📧 ruben.martinezam@alumnos.unican.es  
🎓 Final Bachelor Thesis — Universidad de Cantabria

---
