# 🧠 First-Principles Vertex AI RAG (`first-principles-vertex-rag`)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Vertex AI](https://img.shields.io/badge/Vertex_AI-text--embedding--004-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![Zero Frameworks](https://img.shields.io/badge/Architecture-Zero_Frameworks-10B981?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

Zero-framework **Retrieval-Augmented Generation (RAG)** implementations built from first principles using **Vertex AI Task-Typed Embeddings (`text-embedding-004`)**, pure-Python cosine similarity math $\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$, and **Gemini (`gemini-2.5-flash`)** — without LangChain, LlamaIndex, or external vector databases.

---

## 📐 Architecture

```mermaid
flowchart TD
    Corpus["In-Memory Document Corpus\n(Technical Docs & HR Policies)"] -->|task_type=RETRIEVAL_DOCUMENT| EmbedDoc["Vertex AI text-embedding-004\n(768-dim Document Vectors)"]
    Query["User Question"] -->|task_type=RETRIEVAL_QUERY| EmbedQuery["Vertex AI text-embedding-004\n(768-dim Query Vector)"]
    EmbedDoc --> Cosine["Pure-Python Cosine Similarity\n(a · b) / (||a|| ||b||)"]
    EmbedQuery --> Cosine
    Cosine --> TopK["Top-k Context Selection"]
    TopK --> Gemini["Vertex AI Gemini 2.5 Flash\nStrict Grounded Prompt"]
    Gemini --> Answer["Hallucination-Free Answer"]
```

---

## 📁 Project Structure

```text
first-principles-vertex-rag/
├── main.py                 # Unified CLI entry point (--demo task-typed | hr-policy | all)
├── feature_simple_rag.py   # Asymmetric task-typed RAG (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY)
├── feature_local_rag.py    # HR policy knowledge base RAG pipeline
├── simple_rag.py           # Standalone runner for task-typed RAG
├── local_rag.py            # Standalone runner for HR policy RAG
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `GCP_PROJECT_ID` | Yes | — | Google Cloud Project ID |
| `GCP_LOCATION` | No | `us-central1` | Vertex AI region |
| `EMBEDDING_MODEL` | No | `text-embedding-004` | Vertex AI embedding model |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Vertex AI generative model |

---

## 🚀 Quickstart

```bash
git clone https://github.com/shubhamv3129/first-principles-vertex-rag.git
   cd first-principles-vertex-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set GCP_PROJECT_ID in .env

gcloud auth application-default login
python3 main.py --demo all
```
