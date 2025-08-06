# Insurance Policy Chatbot

This project is an **Insurance Policy Chatbot** built using a **Retrieval-Augmented Generation (RAG)** architecture to answer user questions based on uploaded insurance documents.

The system is powered by:
- **FastAPI** for the backend,
- **Pinecone** as the vector database, and
- **Google Gemini models** for both vector embeddings and conversational responses[^1][^2].

---

## 🧠 Features

- **Document Upload**  
  Supports PDF, DOCX, and TXT file uploads.

- **Text Extraction and Chunking**  
  Uses `pdfplumber` (for PDFs) and `python-docx` (for DOCX) to extract text. The text is then split into semantically coherent chunks.

- **Vector Embeddings**  
  Each text chunk is embedded using the `GoogleGenerativeAIEmbeddings` model (`models/embedding-001`) via the `langchain-google-genai` library[^1].

- **Vector Storage**  
  Embeddings are stored in a Pinecone vector index for fast similarity search.

- **Conversational Q&A**  
  Utilizes a conversational RAG chain that includes context-aware retrieval and generation using the Gemini language model.

- **FastAPI Backend**  
  RESTful endpoints for document upload and chat interface.

- **Simple Frontend**  
  A basic `index.html` interface for interaction.

---

## ⚙️ How It Works

### 1. Document Ingestion (Upload Phase)

1. User uploads a document via the frontend.
2. The FastAPI `upload_document` endpoint receives the file.
3. `document_utils.py` handles:
   - File parsing,
   - Text extraction using `pdfplumber` or `python-docx`,
   - Sentence-level chunking using `nltk`.
4. Each chunk is embedded via `gemini_utils.py` using the Gemini embedding model[^1].
5. Embeddings are upserted into Pinecone via `pinecone_utils.py` along with metadata (filename, content).

### 2. Conversational Q&A (Chat Phase)

1. User submits a question with a `session_id`.
2. The `chat_with_policy` endpoint invokes a **Conversational RAG Chain** built using LangChain[^1]:
   - **History-aware Retriever** reformulates the query using the chat history.
   - **Vector Search** is conducted in Pinecone to fetch top relevant chunks.
   - **LLM Invocation**: The Gemini LLM (`gemini-2.0-flash`) processes the query and retrieved context.
   - **Response Generation**: A clear “Yes” or “No” answer is returned, followed by a short explanation based strictly on document context.
3. The response is sent back to the frontend.

---

## ✅ Prerequisites

Before running the project, ensure you have the following:

1. **Python 3.8 or higher**
2. **Google Gemini API Key**  
   Obtainable from [Google AI Studio](https://aistudio.google.com/app/apikey)[^2]
3. **Pinecone Account**  
   You need an API key and environment name from [Pinecone.io](https://www.pinecone.io)[^2]
4. **Pinecone Index**  
   Create and configure the index name in the `.env` file (default: `insurance-qa-index`)[^2]

---

## 🚀 Local Setup

### Step 1: Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### Step 2: Set Up Environment Variables

Create a `.env` file at the root of the project:

```ini
# --- Google Gemini API Key ---
GEMINI_API_KEY=YOUR_GEMINI_API_KEY

# --- Pinecone Configuration ---
PINECONE_API_KEY=YOUR_PINECONE_API_KEY
PINECONE_ENVIRONMENT=YOUR_PINECONE_ENVIRONMENT
PINECONE_INDEX=insurance-qa-index
```

### Step 3: Install Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

Key libraries include[^1]:
- `fastapi`
- `uvicorn`
- `pinecone-client`
- `pdfplumber`
- `python-docx`
- `google-generativeai`
- `nltk`
- `langchain`, `langchain-google-genai`, `langchain-pinecone`, `langchain-community`

### Step 4: Run the Application

Launch the FastAPI server:

```bash
python main.py
```

By default, the app will run at:  
`http://0.0.0.0:8000`

### Step 5: Access the Chatbot

- Open `http://localhost:8000/api` for API access.
- Visit `http://localhost:8000/docs` for Swagger UI.
- Alternatively, open the `index.html` file in your browser for the chatbot frontend.

---

## 📚 References

[^1]: Implementation libraries and models provided by [LangChain](https://www.langchain.com), [Google Generative AI](https://ai.google.dev), and [Pinecone](https://www.pinecone.io).  
[^2]: API keys and access information can be obtained from the respective provider dashboards.

---
