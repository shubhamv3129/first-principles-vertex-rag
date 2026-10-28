"""
Backwards-compatible runner for simple_rag.py.
"""
from feature_simple_rag import (
    DOCUMENTS,
    ask_gemini,
    cosine_similarity,
    embed_chunks,
    embed_query,
    find_top_chunks,
    make_chunks,
    run_simple_rag,
)


def main() -> None:
    """Executes the Simple RAG demonstration."""
    run_simple_rag()


if __name__ == "__main__":
    main()
