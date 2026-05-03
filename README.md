# RAG PDF Chatbot

Chat with any PDF using AI. Upload a document, ask questions, and get answers based only on your document's content.

🚀 **Live Demo**: https://rag-pdf-chatbot-amaan.streamlit.app/

---

## How It Works

1. Upload any PDF
2. The app extracts and splits text into chunks
3. Each chunk is converted to a vector using Sentence Transformers
4. Vectors are stored in FAISS for fast semantic search
5. Your question is matched against chunks using vector similarity
6. The most relevant chunks are sent to Llama AI to generate an answer

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Web interface |
| PyPDF2 | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector database |
| Groq + Llama 3.3 | LLM for answer generation |

---

## Run Locally

1. Clone the repo
   git clone https://github.com/AmaanShikalgar/rag-pdf-chatbot.git
   cd rag-pdf-chatbot

2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Create .env file
   GROQ_API_KEY=your_key_here

5. Run the app
   streamlit run app.py

---

## Get a Free Groq API Key
https://console.groq.com

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that gives LLMs access
to your specific documents. Instead of relying on training data, the model
retrieves relevant context from your files and answers based only on that.

This is one of the most in-demand skills in GenAI development right now.