# AI-Powered Medical Chatbot with RAG

## Overview

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using domain-specific knowledge retrieved from a medical encyclopedia and generated using OpenAI GPT-4o.

The system combines vector search (Pinecone), embeddings (HuggingFace), and LLM reasoning (GPT-4o) to produce context-aware, grounded responses.

---

## Features

* Retrieval-Augmented Generation (RAG) pipeline
* Medical domain question answering
* Semantic search using Pinecone vector database
* GPT-4o-based response generation
* Document ingestion and chunking pipeline
* Flask-based REST API backend
* Interactive web chat interface
* Modular and extensible architecture

---

## Tech Stack

* Python 3.11
* OpenAI GPT-4o
* LangChain
* Pinecone Vector Database
* HuggingFace Sentence Transformers
* Flask
* Gunicorn
* HTML / Bootstrap

---

## System Architecture

```text
Medical PDF / Encyclopedia
        ↓
Text Chunking
        ↓
HuggingFace Embeddings
        ↓
Pinecone Vector Store
        ↓
User Query
        ↓
Vector Similarity Search (Top-K Retrieval)
        ↓
Context Injection into Prompt
        ↓
GPT-4o Response Generation
        ↓
Flask Web Application
```

---

## Project Structure

```bash
medical-chatbot-with-llm-and-rag/
│
├── app.py
├── store_index.py
├── requirements.txt
├── .env
├── templates/
├── static/
├── src/
│   ├── helper.py
│   ├── prompt.py
│   └── __init__.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/SonasWilson/medical-chatbot-with-llm-and-rag.git
cd medical-chatbot-with-llm-and-rag
```

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

---

## Run the Project

### Step 1: Build Vector Index

```bash
python store_index.py
```

### Step 2: Start Application

```bash
python app.py
```

---

## How It Works

1. Medical documents are loaded and split into chunks
2. Each chunk is converted into embeddings using HuggingFace models
3. Embeddings are stored in Pinecone vector database
4. User query is embedded and matched against stored vectors
5. Top relevant chunks are retrieved
6. Retrieved context is injected into the prompt
7. GPT-4o generates a grounded response
8. Flask serves the response via web interface

---

## Future Improvements

* Streaming responses
* Conversation memory
* Multi-document support
* Docker containerization
* Authentication system
* Advanced agent-based workflow (LangGraph)


<img width="1000" height="813" alt="image" src="https://github.com/user-attachments/assets/80288eab-b770-446b-b333-edd9ad2d1418" />


---

## Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis or treatment.

---

## Author

Sona P Wilson

GitHub: [https://github.com/SonasWilson](https://github.com/SonasWilson)
