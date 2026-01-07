import os
import streamlit as st
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()  


# Streamlit config

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical RAG Chatbot")
st.caption("TinyLlama + Pinecone (Local RAG)")


# Loading embeddings and cached it

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Pinecone retriever (read-only as i have already built the index previously)
@st.cache_resource
def load_retriever():
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        st.error("PINECONE_API_KEY not found. Check your .env file.")
        st.stop()

    pc = Pinecone(api_key=api_key)

    index = pc.Index("medical-chatbot")

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=load_embeddings()
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})


# Local LLM wrapper
class LocalHFLLM(LLM):
    pipeline: any

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output = self.pipeline(
            prompt,
            max_new_tokens=128,   # faster
            temperature=0,
            do_sample=False
        )[0]["generated_text"]
        return output[len(prompt):]

    @property
    def _llm_type(self) -> str:
        return "local_hf"


# Load TinyLlama (Cacheing this and also this is a local model)
@st.cache_resource
def load_llm():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return LocalHFLLM(pipeline=hf_pipeline)


# Building RAG chain and caching it
@st.cache_resource
def build_rag_chain():
    system_prompt = (
        "You are a medical assistant. Use ONLY the provided context to answer. "
        "If the answer is not in the context, say you do not know. "
        "Use at most three concise sentences.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(
        llm=load_llm(),
        prompt=prompt
    )

    return create_retrieval_chain(
        retriever=load_retriever(),
        combine_docs_chain=qa_chain
    )


rag_chain = build_rag_chain()


# UI to input question
query = st.text_input("Ask a medical question:")

if query:
    with st.spinner("Thinking... 🧠"):
        response = rag_chain.invoke({"input": query})

    st.subheader("Answer")
    st.write(response["answer"])
