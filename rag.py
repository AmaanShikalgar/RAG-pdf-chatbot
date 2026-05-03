import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def split_into_chunks(text, chunk_size=500):
    parts = []
    for i in range(0, len(text), 500):
        parts.append(text[i:i+chunk_size])
    return parts

pdf_name = input("Enter PDF filename: ")

if not os.path.exists(pdf_name):
    print("File not found")
    exit()

index_file = pdf_name + ".faiss"
chunks_file = pdf_name + ".pkl"

model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(index_file) and os.path.exists(chunks_file):
    print("Loading from disk...")
    index = faiss.read_index(index_file)
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)
    print("Loaded successfully")
else:
    print("Building index...")
    f = open(pdf_name, "rb")
    reader = PyPDF2.PdfReader(f)
    full_text = ""
    pages = len(reader.pages)
    for i in range(pages):
        page = reader.pages[i]
        text = page.extract_text()
        full_text += text
    chunks = split_into_chunks(full_text)
    vectors = model.encode(chunks)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    print("Built and saved index")

print("Total chunks in FAISS:", index.ntotal)

while True:
    question = input("Ask a question about the document(or 'type' exit to quit): ")
    
    if question == "exit":
        print("Goodbye!")
        break
    
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

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    print("\nAnswer:")
    print(response.choices[0].message.content)