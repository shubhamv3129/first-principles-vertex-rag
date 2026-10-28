"""
Feature module for the in-memory Local RAG demonstration using Vertex AI.
Implements document embedding, pure-Python cosine similarity search,
and grounded answer generation with Gemini.
"""
import math
import os
from typing import Any, Dict, List, Sequence
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
GENERATIVE_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DEFAULT_KNOWLEDGE_BASE: List[str] = [
    "Employees are entitled to 15 days of casual leave and 12 days of sick leave per year.",
    "The standard work-from-home policy allows employees to work remotely up to 3 days a week.",
    "For domestic business travel, such as flights from Delhi to Mumbai, the daily food allowance is capped at ₹2,500.",
    "Maternity leave is provided for 26 weeks, and paternity leave is provided for 2 weeks.",
    "The IT helpdesk is available 24/7. To request a new MacBook charger, raise a ticket via the internal portal.",
]


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """
    Calculates the cosine similarity between two numeric vectors.

    Args:
        v1 (Sequence[float]): First vector.
        v2 (Sequence[float]): Second vector.

    Returns:
        float: Cosine similarity score in [-1.0, 1.0].
    """
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def embed_knowledge_base(
    embedding_model: TextEmbeddingModel, chunks: List[str]
) -> List[Dict[str, Any]]:
    """
    Generates dense vector embeddings for each chunk in the knowledge base.

    Args:
        embedding_model (TextEmbeddingModel): Initialized Vertex AI embedding model.
        chunks (List[str]): List of text strings to embed.

    Returns:
        List[Dict[str, Any]]: List of dicts with 'text' and 'vector' keys.
    """
    document_embeddings: List[Dict[str, Any]] = []
    for chunk in chunks:
        vector = embedding_model.get_embeddings([chunk])[0].values
        document_embeddings.append({"text": chunk, "vector": vector})
    return document_embeddings


def retrieve_top_chunk(
    query_vector: Sequence[float], document_embeddings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Scores all embedded documents against the query vector and returns the top match.

    Args:
        query_vector (Sequence[float]): Embedded user query vector.
        document_embeddings (List[Dict[str, Any]]): Embedded knowledge base items.

    Returns:
        Dict[str, Any]: Highest-scoring document chunk with 'text', 'vector', and 'score'.
    """
    scored_docs = []
    for doc in document_embeddings:
        score = cosine_similarity(query_vector, doc["vector"])
        scored_docs.append({**doc, "score": score})

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[0]


def run_local_rag_demo(
    user_query: str = "How much can I spend on food if I travel to Mumbai for work?",
) -> str:
    """
    Executes the end-to-end in-memory RAG pipeline for a sample HR query.

    Args:
        user_query (str): Question to ask the HR assistant.

    Returns:
        str: Grounded response from Gemini.
    """
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    print("Loading Embedding Model...")
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

    print("Generating vectors for the knowledge base...")
    document_embeddings = embed_knowledge_base(embedding_model, DEFAULT_KNOWLEDGE_BASE)

    print(f"\nUser Query: '{user_query}'\n")
    query_vector = embedding_model.get_embeddings([user_query])[0].values

    print("Searching for relevant chunks...")
    top_doc = retrieve_top_chunk(query_vector, document_embeddings)
    top_chunk = top_doc["text"]
    print(f"-> Top Retrieved Chunk: {top_chunk} (Score: {top_doc['score']:.4f})\n")

    augmented_prompt = f"""System: You are a helpful HR assistant. Answer the user's question using ONLY the provided context below. Do not hallucinate.

Context:
{top_chunk}

Question: {user_query}
Answer:"""

    print("Sending Augmented Prompt to Gemini...")
    llm = GenerativeModel(GENERATIVE_MODEL_NAME)
    response = llm.generate_content(augmented_prompt)
    answer = response.text.strip()

    print("=" * 42)
    print(f"FINAL RAG RESPONSE:\n{answer}")
    print("=" * 42)
    return answer

