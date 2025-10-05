"""Command-line interface (CLI) entry point for RAG-based chatbot and evaluation tool."""

import argparse
from argparse import ArgumentParser

from ragbot.chat import chat
from ragbot.evaluate import evaluate
from ragbot.utils.utils import use_langsmith


def parse_chat_args(subparser: argparse.ArgumentParser):
    """Add command-line arguments for the 'chat' subcommand.

    Args:
        subparser: A subparser object from argparse to attach the chat arguments to.
    """
    subparser.add_argument(
        "-p", "--proj", type=str, required=True, help="Name of the project"
    )
    subparser.add_argument(
        "--llm-provider", type=str, default="google", help="LLM provider"
    )
    subparser.add_argument(
        "--llm", type=str, default="gemini-2.0-flash", help="LLM model"
    )
    subparser.add_argument(
        "--temperature", type=float, default=0.0, help="LLM temperature"
    )
    subparser.add_argument("--top-p", type=float, default=0.85, help="LLM top-p")
    subparser.add_argument("--top-k", type=int, default=40, help="LLM top-k")
    subparser.add_argument(
        "--emb-provider", type=str, default="google", help="Embeddings provider"
    )
    subparser.add_argument(
        "--emb-model",
        type=str,
        default="models/gemini-embedding-001",
        help="Embeddings model",
    )
    subparser.add_argument(
        "--chunk-size", type=int, default=3000, help="Text splitter chunk size"
    )
    subparser.add_argument(
        "--chunk-overlap", type=int, default=600, help="Text splitter chunk overlap"
    )
    subparser.add_argument(
        "--search-type", type=str, default="similarity", help="Retrieval search type"
    )
    subparser.add_argument(
        "--k", type=int, default=4, help="Number of documents to retrieve"
    )
    subparser.set_defaults(func=chat_command)


def parse_evaluate_args(subparser: argparse.ArgumentParser):
    """Add command-line arguments for the 'evaluate' subcommand.

    Args:
        subparser: A subparser object from argparse to attach the evaluate arguments to.
    """
    subparser.add_argument(
        "-p", "--proj", type=str, required=True, help="Name of the project"
    )
    subparser.add_argument(
        "--config-path",
        type=str,
        default="configs/default.json",
        help="Path to config file",
    )
    subparser.add_argument(
        "--dataset-name", type=str, default="ds-test", help="Name of the dataset"
    )
    subparser.set_defaults(func=evaluate_command)


def chat_command(args: argparse.Namespace):
    """Execute the chat command with parsed CLI arguments.

    Args:
        args: Parsed argparse namespace containing chat config.
    """
    chat(
        project_name=args.proj,
        llm_provider=args.llm_provider,
        llm=args.llm,
        llm_temperature=args.temperature,
        llm_top_p=args.top_p,
        llm_top_k=args.top_k,
        embeddings_provider=args.emb_provider,
        embedding_model=args.emb_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        search_type=args.search_type,
        k_docs=args.k,
    )


def evaluate_command(args: argparse.Namespace):
    """Execute the evaluate command with parsed CLI arguments.

    Args:
        args: Parsed argparse namespace containing evaluation config.
    """
    evaluate(
        project_name=args.proj,
        config_path=args.config_path,
        dataset_name=args.dataset_name,
    )


def main():
    """Main CLI entry point.

    Parses command-line arguments and invokes the corresponding function for either
    chat or evaluation mode.
    """
    parser = ArgumentParser(
        prog="poetry run python -m ragbot.cli",
        description=":::A RAG-powered chatbot CLI tool:::",
        usage="poetry run python -m ragbot.cli [-h] <command>",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Use `ragbot.cli <command> --help` for more information on a command.",
    )
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # Subcommand: chat
    chat_parser = subparsers.add_parser(
        "chat",
        help="Run the chat interface",
        usage="poetry run python -m ragbot.cli chat -p <project> [options]",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parse_chat_args(chat_parser)

    # Subcommand: evaluate
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate using a config file",
        usage="poetry run python -m ragbot.cli evaluate -p <project> [options]",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parse_evaluate_args(eval_parser)

    args = parser.parse_args()
    use_langsmith(args.proj)

    try:
        args.func(args)
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1)


if __name__ == "__main__":
    main()
