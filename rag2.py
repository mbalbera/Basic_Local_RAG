import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === Config ===
LOCAL_MODEL_PATH = "./local_model"  # 👈 change this to your actual local path
TOP_K = 1  # Number of retrieved docs to use

# === Load Embedding Model (for document + question embeddings) ===
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# === Load Local Generation Model ===
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# === Sample Corpus ===
documents = [
    "The Eiffel Tower is located in Paris and was completed in 1889.",
    "The capital of Japan is Tokyo, which is also the most populous city.",
    "Python is a programming language known for its simplicity and readability.",
    "The Great Wall of China is one of the longest structures ever built.",
]

# === Create FAISS Vector Store ===
print("[*] Embedding documents...")
doc_embeddings = embedding_model.encode(documents)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))
print(f"[*] FAISS index built with {len(documents)} documents.")

# === RAG Function ===
def rag_query(question: str, top_k: int = TOP_K) -> str:
    print(f"\n[?] Question: {question}")
    
    # Step 1: Embed question
    question_embedding = embedding_model.encode([question])
    
    # Step 2: Retrieve documents
    distances, indices = index.search(np.array(question_embedding), top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    print(f"[*] Retrieved context: {retrieved_docs}")

    # Step 3: Generate answer
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = generator(prompt, max_new_tokens=64, return_full_text=False)[0]['generated_text']
    return response.strip()

# === Try It Out ===
if __name__ == "__main__":
    while True:
        try:
            q = input("\nAsk a question (or type 'exit'): ")
            if q.lower() in ["exit", "quit"]:
                break
            answer = rag_query(q)
            print(f"[✔] Answer: {answer}")
        except KeyboardInterrupt:
            break
