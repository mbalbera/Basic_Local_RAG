# 🧠 Local RAG (Retrieval-Augmented Generation) with Python & Hugging Face

This is a simple implementation of a local Retrieval-Augmented Generation (RAG) pipeline using Python, [FAISS](https://github.com/facebookresearch/faiss) for vector search, and [Hugging Face](https://huggingface.co) models for embeddings and text generation.

## 🔍 What is RAG?

**RAG** combines:
- **Retrieval:** Find relevant documents using a vector database.
- **Augmented Generation:** Use those documents as context to answer questions.

This approach enables more accurate and factual answers, especially for domain-specific data.

---

## 📦 Requirements

Install the dependencies:

```bash
pip install transformers sentence-transformers faiss-cpu
````
🛠️ How It Works
Document Embedding: A list of documents is embedded using sentence-transformers.

Vector Store: These embeddings are stored in a FAISS index for efficient similarity search.

Retrieval: When a user asks a question, the top-k similar documents are retrieved.

Generation: A Hugging Face text generation model (e.g., flan-t5-base) answers the question using the retrieved context.

🚀 Example Usage
```python
response = rag_query("Where is the Eiffel Tower located?")
print(response)
```
Output:

The Eiffel Tower is located in Paris.
📁 File Structure
bash
Copy
Edit
├── rag.py            # Main RAG implementation
├── README.md         # This file


🧪 Sample Documents Used

documents = [
    "The Eiffel Tower is located in Paris and was completed in 1889.",
    "The capital of Japan is Tokyo, which is also the most populous city.",
    "Python is a programming language known for its simplicity and readability.",
    "The Great Wall of China is one of the longest structures ever built.",
]

🔄 To-Do / Improvements
 Add support for loading documents from .txt or .pdf

 Persist and reload FAISS index

 Swap in a more powerful embedding model (e.g., BAAI/bge-base-en)

 Switch to a local large language model (e.g., Mistral, LLaMA)

💡 Credits

FAISS

Hugging Face Transformers

SentenceTransformers

📝 License
MIT License. Use freely for learning and experimentation!