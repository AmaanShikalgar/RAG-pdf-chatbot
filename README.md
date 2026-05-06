# 📄 DocuMind AI

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0ea5e9,100:1e293b&height=200&section=header&text=DocuMind%20AI&fontSize=40&fontColor=ffffff&animation=fadeIn" />
</p>

<p align="center">
AI-powered Document Question Answering System using Retrieval-Augmented Generation (RAG)
</p>

---

## 📸 Live Demo
https://rag-pdf-chatbot-amaan.streamlit.app/

---

## ⚡ Features

- Chat with any PDF document  
- Semantic search using FAISS  
- Embedding generation via Sentence Transformers  
- Context-aware responses using LLaMA 3 (Groq API)  
- Page-level source tracking  
- Chat history support  
- Streamlit interactive UI  

---

## 🧠 How It Works

1. Upload PDF  
2. Extract text  
3. Split into chunks  
4. Convert chunks into embeddings  
5. Store embeddings in FAISS index  
6. Convert query into embedding  
7. Retrieve most relevant chunks  
8. Send context to LLM  
9. Generate grounded answer  

---

## 🏗️ Architecture

PDF → Text Extraction → Chunking → Embeddings → FAISS Index → Retrieval → LLM (LLaMA 3) → Answer

---

## 🛠️ Tech Stack

**Backend / AI:**
- Python  
- FAISS  
- Sentence Transformers  
- PyPDF2  
- Groq API (LLaMA 3.3)

**Frontend:**
- Streamlit  

**Concept:**
- Retrieval-Augmented Generation (RAG)

---

## 📁 Project Structure

DocuMind-AI/  
├── app.py  
├── requirements.txt  
├── .env (ignored)  
├── README.md  
└── screenshots/  
&nbsp;&nbsp;&nbsp;&nbsp;└── demo.png  

---

## 🚀 Run Locally

### Clone repository
git clone https://github.com/your-username/DocuMind-AI.git  
cd DocuMind-AI  

### Create virtual environment
python -m venv venv  
venv\Scripts\activate  

### Install dependencies
pip install -r requirements.txt  

### Add environment variables
GROQ_API_KEY=your_api_key_here  

### Run app
streamlit run app.py  

---

## 🔑 API Key
https://console.groq.com

---

## 📚 What is RAG?

Retrieval-Augmented Generation (RAG) is a technique where:

- Relevant information is retrieved from documents  
- That context is given to an LLM  
- The model generates grounded responses  

This reduces hallucination and improves factual accuracy.

---

## 💡 Key Learnings

- Vector embeddings and semantic search  
- FAISS-based retrieval systems  
- LLM integration with external context  
- Prompt engineering for grounding responses  
- End-to-end AI system design  

---

## 👨‍💻 Author

**Amaan Shikalgar**  
Full Stack & AI Developer  

---

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub.
