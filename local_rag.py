"""
Backwards-compatible runner for the Local RAG script.
"""
from feature_local_rag import cosine_similarity, run_local_rag_demo

__all__ = ["cosine_similarity", "run_local_rag_demo"]

if __name__ == "__main__":
    run_local_rag_demo()
