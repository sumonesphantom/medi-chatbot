# Medical RAG Chatbot

This project is a local Retrieval-Augmented Generation (RAG) chatbot built for medical question answering using a single reference book, this is in the data folder. The system retrieves relevant text from a Pinecone vector database and generates grounded responses using a locally hosted TinyLlama model.

## Features

- Fully local LLM inference using TinyLlama 1.1B
- Retrieval-Augmented Generation with Pinecone
- Sentence-Transformers embeddings
- Streamlit chat interface
- Strict context grounding (answers only from the indexed book)
- Cached models and embeddings for performance

## Architecture Overview

1. Medical book is ingested and split into chunks
2. Embeddings are generated using all-MiniLM-L6-v2
3. Vectors are stored in Pinecone with book metadata
4. User queries are embedded and matched against the book
5. Retrieved context is passed to TinyLlama for answer generation

## Requirements

Python 3.9 or higher is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
