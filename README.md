<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Qdrant-DC382D?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant"/>
</p>

# 🏥 Medical RAG QA — Meditron 7B LLM

> A **Retrieval-Augmented Generation (RAG)** powered medical question-answering system that uses **Meditron 7B** LLM, **Qdrant** vector database, and **PubMedBERT** embeddings to deliver accurate, document-grounded answers to medical and oncology questions.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Configuration](#%EF%B8%8F-configuration)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔍 Overview

Standard LLMs often **hallucinate** medical facts, producing confident but incorrect information — a serious risk in healthcare. This project solves that problem by implementing a **RAG pipeline** that grounds every answer in **real medical PDF documents**.

Users ask medical questions through a web-based chat interface, and the system retrieves relevant context from a vector database of ingested medical documents before generating an answer using a **locally-running, privacy-preserving medical LLM**.

### Why This Approach?

| Problem | Solution |
|---|---|
| LLMs hallucinate medical facts | RAG grounds answers in **real medical documents** |
| Medical data is in unstructured PDFs | Pipeline **extracts, chunks, and vectorizes** PDF content |
| Cloud APIs are expensive & raise privacy concerns | **Locally-running open-source LLM** (Meditron 7B via GGUF) |
| General embeddings miss medical semantics | **PubMedBERT** embeddings trained on biomedical literature |
| No user-friendly interface | **Web-based chat UI** via FastAPI + Jinja2 + Bootstrap 5 |

---

## ✨ Key Features

- 🧠 **Medical-domain LLM** — Meditron 7B, fine-tuned on clinical guidelines & PubMed data
- 🔒 **Privacy-first** — Runs entirely locally, no data sent to external APIs
- 📄 **PDF knowledge base** — Ingest any medical PDF as a knowledge source
- ⚡ **CPU inference** — Quantized GGUF model runs without a GPU
- 🎯 **Source attribution** — Every answer includes the source document and context
- 🖥️ **Web chat UI** — Dark-themed, responsive interface built with Bootstrap 5
- 🔍 **Semantic search** — PubMedBERT embeddings for biomedical-aware retrieval
- 🗄️ **Production-grade vector DB** — Qdrant for fast, scalable similarity search

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER (Browser)                       │
│              http://localhost:8000                        │
└──────────────────────┬──────────────────────────────────┘
                       │  HTTP POST /get_response
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Server (rag.py)                 │
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │  Jinja2 UI   │   │ Prompt       │   │ RetrievalQA  │  │
│  │ (index.html) │   │ Template     │   │   Chain       │  │
│  └─────────────┘   └──────────────┘   └──────┬───────┘  │
│                                              │           │
└──────────────────────────────────────────────┼───────────┘
                       │                       │
          ┌────────────┘                       │
          ▼                                    ▼
┌──────────────────┐              ┌──────────────────────┐
│  Qdrant Vector   │◄─── query ──│  PubMedBERT          │
│  Database        │   embedding  │  Embeddings          │
│  (localhost:6333)│              │  (NeuML/pubmedbert)  │
└──────────────────┘              └──────────────────────┘
          │
          │  top-k context
          ▼
┌──────────────────────────────────────────────────────────┐
│              Meditron 7B LLM (GGUF, local)               │
│          Loaded via CTransformers (llama type)           │
│                                                          │
│  Config: max_tokens=1024, temp=0.1, top_k=50, top_p=0.9 │
└──────────────────────────────────────────────────────────┘
          │
          ▼
    Generated Answer + Source Document → Browser
```

---

## 🛠 Tech Stack

### Core AI/ML

| Technology | Role |
|---|---|
| **[Meditron 7B](https://huggingface.co/epfl-llm/meditron-7b)** | Medical LLM (GGUF Q4_K_M quantized) for answer generation |
| **[PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings)** | Biomedical embedding model for text vectorization |
| **[CTransformers](https://github.com/marella/ctransformers)** | CPU-optimized runtime for GGUF quantized models |
| **[LangChain](https://www.langchain.com/)** | RAG orchestration framework (RetrievalQA chain) |

### Data & Storage

| Technology | Role |
|---|---|
| **[Qdrant](https://qdrant.tech/)** | High-performance vector database for embeddings |
| **PyPDFLoader** | PDF text extraction from medical documents |
| **RecursiveCharacterTextSplitter** | Document chunking with overlap for context preservation |

### Web & API

| Technology | Role |
|---|---|
| **[FastAPI](https://fastapi.tiangolo.com/)** | Async Python web framework |
| **Jinja2** | Server-side HTML templating |
| **Bootstrap 5** | Responsive dark-themed UI |
| **Uvicorn** | ASGI server |

---

## 📁 Project Structure

```
Medical-RAG-using-Meditron-7B-LLM/
│
├── Data/                           # Medical PDF knowledge base
│   ├── cancer_and_cure__a_critical_analysis.27.pdf
│   └── medical_oncology_handbook_june_2020_edition.pdf
│
├── templates/
│   └── index.html                  # Web chat interface (Bootstrap 5)
│
├── ingest.py                       # Step 1: PDF ingestion → Vector DB
├── rag.py                          # Step 2: FastAPI server + RAG pipeline
├── retriever.py                    # Utility: standalone retrieval testing
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Docker** (for Qdrant) or Qdrant installed locally
- **~4 GB disk space** for the quantized model file

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Medical-RAG-using-Meditron-7B-LLM.git
cd Medical-RAG-using-Meditron-7B-LLM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Meditron 7B Model

Download the quantized GGUF model file from Hugging Face and place it in the project root:

```bash
# Download meditron-7b.Q4_K_M.gguf from:
# https://huggingface.co/TheBloke/meditron-7B-GGUF
```

### 4. Start Qdrant Vector Database

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Ingest Medical Documents

```bash
python ingest.py
```

This reads all PDFs from `Data/`, generates PubMedBERT embeddings, and stores them in Qdrant.

### 6. Run the Application

```bash
uvicorn rag:app --reload
```

Open **http://localhost:8000** in your browser.

---

## 💬 Usage

1. Open the web interface at `http://localhost:8000`
2. Type a medical question in the text area (e.g., _"What is Metastatic disease?"_)
3. Click **Submit**
4. The system returns:
   - ✅ **Answer** — Generated by Meditron 7B, grounded in your documents
   - 📄 **Source Context** — The exact document chunk used for the answer
   - 📁 **Source Document** — The PDF file the information came from

---

## ⚙ How It Works

The application operates in **two phases**:

### Phase 1 — Document Ingestion (`ingest.py`)

```
Medical PDFs (Data/)
    → Load with PyPDFLoader
    → Split into chunks (1000 chars, 100 char overlap)
    → Generate embeddings with PubMedBERT
    → Store vectors in Qdrant DB
```

- **Chunk size of 1000** balances context richness with retrieval precision
- **100-character overlap** prevents information loss at chunk boundaries
- **PubMedBERT** embeddings capture biomedical semantic relationships (e.g., _"neoplasm" ≈ "tumor"_)

### Phase 2 — Query & Answer (`rag.py`)

```
User Question (Web UI)
    → Embed question with PubMedBERT
    → Similarity search in Qdrant (top-1 result)
    → Retrieved context + question → Prompt Template
    → Meditron 7B generates answer
    → Return answer + source to UI
```

- **Low temperature (0.1)** ensures deterministic, factual responses
- Prompt explicitly instructs the model to **not hallucinate** — if the answer isn't in the context, it says "I don't know"
- **Source attribution** enables users to verify every answer

---

## ⚙️ Configuration

The LLM configuration in `rag.py` can be tuned:

| Parameter | Default | Description |
|---|---|---|
| `max_new_tokens` | 1024 | Maximum tokens in generated response |
| `context_length` | 2048 | Context window size |
| `temperature` | 0.1 | Lower = more deterministic (recommended for medical) |
| `top_k` | 50 | Top-K sampling parameter |
| `top_p` | 0.9 | Nucleus sampling parameter |
| `repetition_penalty` | 1.1 | Penalizes repeated tokens |
| `threads` | CPU cores / 2 | Number of inference threads |

---
