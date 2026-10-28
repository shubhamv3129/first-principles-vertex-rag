"""
Main entry point for the First-Principles Vertex AI RAG reference module.
Demonstrates both asymmetric task-typed embeddings (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY)
and in-memory HR policy RAG using pure-Python cosine similarity math.
"""
import argparse
from feature_local_rag import run_local_rag_demo
from feature_simple_rag import run_simple_rag


def example_task_typed_rag() -> None:
    """
    Runs the multi-question Vertex AI RAG pipeline using RETRIEVAL_DOCUMENT
    and RETRIEVAL_QUERY task types.
    """
    run_simple_rag()


def example_hr_policy_rag() -> None:
    """
    Runs the HR travel & leave policy in-memory RAG pipeline.
    """
    run_local_rag_demo()


def main() -> None:
    """
    CLI entry point supporting `--demo task-typed`, `--demo hr-policy`, or `--demo all`.
    """
    parser = argparse.ArgumentParser(
        description="First-Principles Vertex AI RAG Reference Implementations"
    )
    parser.add_argument(
        "--demo",
        choices=["task-typed", "hr-policy", "all"],
        default="all",
        help="Which RAG demonstration to execute (default: all).",
    )
    args = parser.parse_args()

    if args.demo in ("task-typed", "all"):
        print("=== 1. Asymmetric Task-Typed Vertex AI RAG ===")
        example_task_typed_rag()

    if args.demo in ("hr-policy", "all"):
        print("\n=== 2. HR Policy In-Memory RAG ===")
        example_hr_policy_rag()


if __name__ == "__main__":
    main()
