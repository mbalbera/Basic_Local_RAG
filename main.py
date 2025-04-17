import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Step 1: Load Embedding and Generation Models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, good quality
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Step 2: Sample Documents to Build Vector Store
documents = [
    "The Eiffel Tower is located in Paris and was completed in 1889.",
    "The capital of Japan is Tokyo, which is also the most populous city.",
    "Python is a programming language known for its simplicity and readability.",
    "The Great Wall of China is one of the longest structures ever built.",
]

# Step 3: Create Embeddings and Store with FAISS
doc_embeddings = embedding_model.encode(documents)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Step 4: Retrieval + Generation (RAG loop)
def rag_query(question, top_k=1):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    
    retrieved_docs = [documents[i] for i in indices[0]]
    context = " ".join(retrieved_docs)

    # Prompt: You can customize this
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = generator(prompt, max_new_tokens=64, return_full_text=False)[0]['generated_text']
    return response

# Try it!
print(rag_query("Where is the Eiffel Tower located?"))
