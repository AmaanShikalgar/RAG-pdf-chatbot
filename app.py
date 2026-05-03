import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from groq import Groq
from dotenv import load_dotenv
import io

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def split_into_chunks(text, chunk_size=500):
    parts = []
    for i in range(0, len(text), 500):
        parts.append(text[i:i+chunk_size])
    return parts

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    if "chunks" not in st.session_state:
        with st.spinner("Reading and indexing your PDF..."):
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            full_text = ""
            for i in range(len(reader.pages)):
                full_text += reader.pages[i].extract_text()
            chunks = split_into_chunks(full_text)
            vectors = model.encode(chunks)
            dimension = vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(vectors)
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["history"] = []
        st.success(f"PDF indexed. {len(chunks)} chunks ready.")

    # show chat history
    # show chat history
    for chat in st.session_state.get("history", []):
        st.write(f"**You:** {chat['question']}")
        st.write(f"**Answer:** {chat['answer']}")
        st.divider()

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input("Ask a question (or type 'exit' to clear chat):")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        if question.lower() == "exit":
            st.session_state["history"] = []
            st.rerun()
        else:
            index = st.session_state["index"]
            chunks = st.session_state["chunks"]
            question_vector = model.encode([question])
            distances, indices = index.search(question_vector, k=3)
            context = ""
            for i in indices[0]:
                context += chunks[i] + "\n\n"
            prompt = f"""Answer the question based only on the context below.
If the answer is not in the context say "I cannot find this in the document."

Context:
{context}

Question:
{question}

Answer:"""
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
            answer = response.choices[0].message.content
            st.session_state["history"].append({
                "question": question,
                "answer": answer
            })
            st.rerun()