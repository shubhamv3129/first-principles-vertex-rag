"""
Feature module for the Simple Vertex AI RAG demonstration.
Uses Vertex AI task-typed embeddings (RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY),
pure-Python cosine similarity scoring, and Gemini grounded Q&A.
"""
import math
import os
from typing import Any, Dict, List, Sequence
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
GENERATIVE_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DOCUMENTS: List[Dict[str, str]] = [
    {
        "title": "Vertex AI overview",
        "text": (
            "Vertex AI is Google Cloud's unified platform for machine learning. "
            "It lets you build, deploy, and scale ML models and AI applications. "
            "Vertex AI includes AutoML for no-code training and Custom Training "
            "for full control. The Model Garden hosts over 150 foundation models "
            "including Gemini, PaLM, and Llama."
        ),
    },
    {
        "title": "Vertex AI Embeddings",
        "text": (
            "Vertex AI offers the text-embedding-004 model which produces 768-dimensional "
            "vectors. It supports task types: RETRIEVAL_DOCUMENT for corpus texts and "
            "RETRIEVAL_QUERY for search queries. These task types improve retrieval "
            "accuracy because the model is optimised differently for each role. "
            "The API accepts up to 250 texts per batch request."
        ),
    },
    {
        "title": "Vertex AI Gemini",
        "text": (
            "Gemini is Google's multimodal foundation model available on Vertex AI. "
            "gemini-2.5-flash is fast and cost-efficient, suited for high-volume tasks. "
            "Gemini accepts text, images, audio, and video as input. "
            "You can control output behaviour with temperature, top_p, and max_output_tokens."
        ),
    },
    {
        "title": "What is RAG",
        "text": (
            "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
            "generative model. First, you retrieve relevant text from a knowledge base "
            "using semantic search. Then you pass that text as context to an LLM so it "
            "can answer grounded in real information rather than relying on training memory. "
            "RAG reduces hallucination and keeps answers up to date."
        ),
    },
]


def make_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Converts raw document records into structured chunk dictionaries.

    Args:
        documents (List[Dict[str, str]]): List of documents with 'title' and 'text'.

    Returns:
        List[Dict[str, Any]]: List of chunk records.
    """
    chunks = [
        {
            "id": doc["title"],
            "source": doc["title"],
            "text": doc["text"],
        }
        for doc in documents
    ]
    print(f"[1] Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def embed_chunks(
    chunks: List[Dict[str, Any]], model: TextEmbeddingModel
) -> List[Dict[str, Any]]:
    """
    Batch-embeds document chunks using the RETRIEVAL_DOCUMENT task type.

    Args:
        chunks (List[Dict[str, Any]]): Document chunks to embed.
        model (TextEmbeddingModel): Initialized Vertex AI embedding model.

    Returns:
        List[Dict[str, Any]]: Chunks populated with an 'embedding' vector field.
    """
    print(f"[2] Embedding {len(chunks)} chunks with {EMBEDDING_MODEL_NAME} ...")
    inputs = [
        TextEmbeddingInput(text=c["text"], task_type="RETRIEVAL_DOCUMENT")
        for c in chunks
    ]
    results = model.get_embeddings(inputs)
    for chunk, result in zip(chunks, results):
        chunk["embedding"] = result.values
    print(f"    Done. Each embedding has {len(chunks[0]['embedding'])} dimensions.")
    return chunks


def embed_query(query: str, model: TextEmbeddingModel) -> List[float]:
    """
    Embeds a search query using the RETRIEVAL_QUERY task type.

    Args:
        query (str): User search query.
        model (TextEmbeddingModel): Initialized Vertex AI embedding model.

    Returns:
        List[float]: 768-dimensional embedding vector.
    """
    print(f"[3] Embedding query: '{query}'")
    result = model.get_embeddings(
        [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
    )
    return result[0].values


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """
    Computes cosine similarity between two numeric vectors.

    Args:
        vec_a (Sequence[float]): First vector.
        vec_b (Sequence[float]): Second vector.

    Returns:
        float: Similarity score between -1.0 and 1.0.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def find_top_chunks(
    query_embedding: Sequence[float],
    chunks: List[Dict[str, Any]],
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    """
    Ranks all chunks by cosine similarity to `query_embedding` and returns the top-k.

    Args:
        query_embedding (Sequence[float]): Embedded query vector.
        chunks (List[Dict[str, Any]]): Embedded document chunks.
        top_k (int): Number of top chunks to return.

    Returns:
        List[Dict[str, Any]]: Top-k matching chunks.
    """
    print(f"[4] Scoring {len(chunks)} chunks by cosine similarity ...")
    scored = [
        (cosine_similarity(query_embedding, chunk["embedding"]), chunk)
        for chunk in chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    print("    Scores:")
    for score, chunk in scored:
        print(f"      {score:.4f}  {chunk['source']}")

    return [chunk for _, chunk in scored[:top_k]]


def ask_gemini(
    query: str,
    top_chunks: List[Dict[str, Any]],
    gemini_model: GenerativeModel,
) -> str:
    """
    Prompts Gemini to answer `query` grounded strictly in `top_chunks`.

    Args:
        query (str): User question.
        top_chunks (List[Dict[str, Any]]): Retrieved context chunks.
        gemini_model (GenerativeModel): Initialized Gemini model.

    Returns:
        str: Grounded response text.
    """
    context = "\n\n".join(
        f"[{i + 1}] {c['source']}:\n{c['text']}" for i, c in enumerate(top_chunks)
    )
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {query}
Answer:"""

    print("[5] Sending context + question to Gemini ...")
    response = gemini_model.generate_content(prompt)
    return response.text


def run_simple_rag() -> None:
    """
    Initializes Vertex AI, builds the in-memory embedded index, and answers
    sample questions using grounded retrieval.
    """
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    gemini_model = GenerativeModel(GENERATIVE_MODEL_NAME)

    chunks = make_chunks(DOCUMENTS)
    chunks = embed_chunks(chunks, embedding_model)

    questions = [
        "What task types does text-embedding-004 support?",
        "How does RAG reduce hallucination?",
        "What models are available in the Vertex AI Model Garden?",
    ]

    for question in questions:
        print("\n" + "─" * 60)
        print(f"QUESTION: {question}")
        print("─" * 60)

        query_embedding = embed_query(question, embedding_model)
        top_chunks = find_top_chunks(query_embedding, chunks, top_k=2)
        answer = ask_gemini(question, top_chunks, gemini_model)

        print(f"\nANSWER:\n{answer}")

