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

load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="Medical RAG Chatbot",
    layout="centered"
)

st.title("Medical RAG Chatbot")
st.caption("TinyLlama with Pinecone Retrieval")

# Sidebar
with st.sidebar:
    st.header("System Information")
    st.write("Model: TinyLlama 1.1B Chat")
    st.write("Embedding: all-MiniLM-L6-v2")
    st.write("Vector DB: Pinecone")
    st.divider()
    show_context = st.checkbox("Show retrieved context", value=False)

# Resources
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_retriever():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("PINECONE_API_KEY not found.")
        st.stop()

    pc = Pinecone(api_key=api_key)
    index = pc.Index("medical-chatbot")

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=load_embeddings()
    )

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"source": "medical_book"}  
        }
    )

class LocalHFLLM(LLM):
    pipeline: any

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output = self.pipeline(
            prompt,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False
        )[0]["generated_text"]

        return output[len(prompt):]

    @property
    def _llm_type(self) -> str:
        return "local_hf"

@st.cache_resource
def load_llm():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return LocalHFLLM(pipeline=pipe)

@st.cache_resource
def build_rag_chain():
    system_prompt = (
        "You are a medical assistant.\n"
        "Answer ONLY using the provided book context.\n"
        "Do NOT use prior knowledge.\n"
        "If the answer is not present in the book, say \"I do not know based on the provided text.\" \n"
        "Limit your answer to a maximum of three concise sentences.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    qa_chain = create_stuff_documents_chain(
        llm=load_llm(),
        prompt=prompt
    )

    return create_retrieval_chain(
        retriever=load_retriever(),
        combine_docs_chain=qa_chain
    )

rag_chain = build_rag_chain()

# Chat State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask a medical question based on the book")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Generating answer"):
        response = rag_chain.invoke({"input": user_input})

    answer = response["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    if show_context:
        with st.expander("Retrieved Context"):
            for doc in response["context"]:
                st.write(doc.page_content)
