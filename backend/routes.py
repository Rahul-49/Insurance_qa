# routes.py
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import logging
from dotenv import load_dotenv

from document_utils import process_and_embed_document
from pinecone_utils import get_pinecone_retriever
from chain_builder import get_conversational_rag_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
# --- ADDED: Import for the embedding model object ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Setup ---
router = APIRouter()
load_dotenv() # Ensure env variables are loaded
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- In-Memory Session Store ---
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Gets a chat history object for a given session_id."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Build Conversational Chain ---
# This is created once when the server starts.
# --- FIXED: Create the LangChain embedding model object correctly ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
retriever = get_pinecone_retriever(embeddings_model)
# -----------------------------------------------------------------
conversational_rag_chain = get_conversational_rag_chain(retriever, get_session_history)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    session_id: str

class UploadResponse(BaseModel):
    message: str
    chunks_indexed: int
    filename: str

# --- API Endpoints ---
@router.get("/", summary="Health Check")
async def root():
    return {"message": "Insurance QA System API is running and healthy"}

@router.post("/upload", response_model=UploadResponse, summary="Upload and Process a Document")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        logger.info(f"--- UPLOAD START: {file.filename} ---")
        contents = await file.read()
        chunks_indexed = process_and_embed_document(contents, file.filename)
        logger.info(f"--- UPLOAD SUCCESS: {file.filename} ---")
        return {
            "message": f"Document '{file.filename}' processed and indexed successfully.",
            "chunks_indexed": chunks_indexed,
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"--- UPLOAD FAILED: {file.filename} ---", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/chat", summary="Chat with Policy Documents (Conversational)")
async def chat_with_policy(request: ChatRequest):
    """
    Main conversational endpoint. It requires a session_id to maintain chat history.
    """
    query = request.query
    session_id = request.session_id
    logger.info(f"--- CHAT START (Session: {session_id}): Received query: '{query}' ---")
    
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        response = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}}
        )
        
        logger.info(f"--- CHAT SUCCESS (Session: {session_id}) ---")
        return {"answer": response["answer"]}

    except Exception as e:
        logger.error(f"--- CHAT FAILED (Session: {session_id}) ---", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")