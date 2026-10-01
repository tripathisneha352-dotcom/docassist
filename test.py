
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pymupdf

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "The 'sentence-transformers' package is not installed. "
        "Install it with: pip install sentence-transformers"
    ) from exc

pdf_path = "bank.pdf"

document = pymupdf.open(pdf_path)

text = ""

for page in document:
    text += page.get_text()

document.close()

words = text.split()

chunk_size = 1000
overlap = 100

chunks = []

for i in range(0, len(words), chunk_size):
    chunk = " ".join(words[i:i + chunk_size - overlap])
    chunks.append(chunk)



model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

import faiss
import numpy as np

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Total vectors stored:", index.ntotal)

question = "What documents are required to open a bank account?"

# 1. Convert question to embedding
question_embedding = model.encode([question])
question_embedding = np.array(question_embedding).astype("float32")

# 2. Retrieve relevant chunks
distances, indices = index.search(question_embedding, k=3)

# 3. Combine retrieved chunks
context = ""

for i in indices[0]:
    context += chunks[i] + "\n\n"

# 4. Send context + question to LLM
prompt = f"Answer the question using the context below.\n\nContext:\n{context}\nQuestion: {question}"
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt
)

print("\nAnswer:")
print(response.text)























