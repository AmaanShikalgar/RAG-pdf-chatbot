import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv
import io

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def split_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("Chat with your PDF")

if "history" not in st.session_state:
    st.session_state["history"] = []

st.divider()

if st.button("Clear Chat"):
    st.session_state["history"] = []
    st.rerun()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.session_state["uploaded_file"] = uploaded_file

if "uploaded_file" in st.session_state:
    uploaded_file = st.session_state["uploaded_file"]

    if "chunks" not in st.session_state:
        with st.spinner("Reading and indexing your PDF..."):
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))

            chunks = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    for chunk in split_into_chunks(text):
                        chunks.append({
                            "text": chunk,
                            "page": i + 1
                        })

            texts = [c["text"] for c in chunks]

            vectors = model.encode(texts)
            dimension = vectors.shape[1]

            index = faiss.IndexFlatL2(dimension)
            index.add(vectors)

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["history"] = []

        st.success(f"PDF indexed. {len(chunks)} chunks ready.")

    for chat in st.session_state.get("history", []):
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    question = st.chat_input("Ask something about your PDF...")

    if question:
        index = st.session_state["index"]
        chunks = st.session_state["chunks"]

        question_vector = model.encode([question])

        distances, indices = index.search(question_vector, k=5)

        context = ""
        sources = set()

        for i in indices[0]:
            chunk = chunks[i]
            context += f"(Page {chunk['page']}) {chunk['text']}\n\n"
            sources.add(chunk["page"])

        prompt = f"""
You are a precise assistant.

Rules:
- Answer ONLY from the provided context
- If the answer is not in the context, say: "I cannot find this in the document."
- Keep the answer concise
- Mention page numbers if relevant

Context:
{context}

Question:
{question}

Answer:
"""

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

        answer = response.choices[0].message.content

        if sources:
            answer += f"\n\nSources: Pages {', '.join(map(str, sorted(sources)))}"

        st.session_state["history"].append({
            "question": question,
            "answer": answer
        })

        st.rerun()
       
st.markdown("""
<style>
.footer {
    text-align: center;
    font-size: 14px;
    color: gray;
    margin-top: 50px;
    padding-bottom: 20px;
}
</style>

<div class="footer">
© 2026 Amaan Shikalgar • 
<a href="https://github.com/your-username/your-repo" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)