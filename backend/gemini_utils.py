# gemini_utils.py
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialization ---
load_dotenv()

# Google Gemini client configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in the environment.")
genai.configure(api_key=GEMINI_API_KEY)

# --- Core Functions ---
def get_embeddings(texts: list, model="models/embedding-001") -> list:
    """
    Generates embeddings for a list of texts using the Gemini API.
    """
    if not texts or not all(isinstance(t, str) for t in texts):
        logger.warning("get_embeddings called with invalid input.")
        return []
    try:
        result = genai.embed_content(
            model=model,
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Error getting embeddings from Gemini: {e}")
        raise

def get_decision_from_llm(query: str, context_chunks: list) -> dict:
    """
    Generates a structured decision using Gemini, based on the query and retrieved context.
    Includes a fallback when no context is found.
    """
    if not context_chunks:
        logger.warning("No relevant context found in Pinecone. Falling back to heuristic response.")
        # This is a helpful fallback you designed for when no context is found.
        return {
            "Decision": "Indeterminate",
            "Amount": 0,
            "Justification": {
                "summary": "No matching clauses were found in the provided policy documents to make a decision. Please try rephrasing your question with more specific terms found in the policy.",
                "clauses": [],
                "suggested_followup": "Is the knee problem caused by an accident or was it a pre-existing condition before the policy started? Using terms like 'accident' or 'pre-existing' might yield better results."
            }
        }

    # Assemble the context from the retrieved chunks
    context = "\n---\n".join([chunk['metadata']['text'] for chunk in context_chunks])

    # Construct the prompt
    prompt = f"""
You are an expert AI assistant for processing insurance claims. Your task is to act as a policy adjudicator.
Automatically understand whether user is trying to talk causually or related to insurance.
Carefully analyze the user's query and the provided policy clauses.
You MUST base your decision ONLY on the information contained within the provided policy clauses. Do not use any external knowledge.

Here are the relevant policy clauses:
---
{context}
---

Based on these clauses, please adjudicate the following user query:
User Query: "{query}"

Your final output MUST be a single, valid JSON object with the following structure:
{{
  "Decision": "Approved" | "Rejected" | "Indeterminate",
  "Amount": <integer>,
  "Justification": {{
    "summary": "<string: A concise, informative explanation in atmost 2 sentences for your decision.>",
    "clauses": [
      {{
        "clause_number": "<string: The specific clause number, e.g., 'D.2.f.27'. If not available, use 'N/A'>",
        "text": "<string: The exact, verbatim text from the policy clause that supports your decision.>",
        "source_document": "<string: The source document name.>",
        "page_number": "<integer: The page number where the clause is found. If not available, use 0>"
      }}
    ]
  }}
}}
"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        decision_json = json.loads(response.text)
        logger.info(f"Gemini generated decision: {decision_json.get('Decision')}")
        return decision_json

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Gemini response: {e.text}")
        raise ValueError("The model returned an invalid JSON object.")
    except Exception as e:
        logger.error(f"Error getting decision from Gemini: {e}")
        raise
