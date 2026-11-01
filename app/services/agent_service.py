
import os
import re
from pathlib import Path
from typing import Generator, List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.utils.vector_db import load_vectorstore
from app.services.ml_service import predict_confidence
from app.tools.ticket_tool import create_ticket
from app.tools.refund_tool import process_refund
from app.tools.order_tool import get_order_status


# Load .env from the project root (lumenAI-main/.env).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

_VECTORSTORE = None

DEFAULT_GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)
DEFAULT_GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1",
)


def get_vectorstore():
    """Load and cache the FAISS vector store."""
    global _VECTORSTORE

    if _VECTORSTORE is not None:
        return _VECTORSTORE

    _VECTORSTORE = load_vectorstore()
    return _VECTORSTORE


def _distance_to_similarity(distance: float) -> float:
    """Convert a non-negative distance into a similarity score."""
    safe_distance = max(0.0, float(distance))
    return 1.0 / (1.0 + safe_distance)


def retrieve_ranked_documents(
    query: str,
    k: int = 3,
) -> List[Tuple[object, float]]:
    """Retrieve documents and rank them by similarity."""
    vectorstore = get_vectorstore()

    try:
        matches = vectorstore.similarity_search_with_score(query, k=k)
    except AttributeError:
        docs = vectorstore.similarity_search(query, k=k)
        return [(doc, 0.5) for doc in docs]

    return [
        (doc, _distance_to_similarity(score))
        for doc, score in matches
    ]


def _get_chat_model() -> ChatOpenAI:
    """Initialize the Groq chat model through the OpenAI-compatible API."""
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise FileNotFoundError(
            "GROQ_API_KEY is missing. Add it to your project-root .env file."
        )

    if groq_api_key.strip() == "your_groq_api_key_here":
        raise FileNotFoundError(
            "GROQ_API_KEY is a placeholder. Replace it with your actual Groq API key."
        )

    # Read configuration at call time so environment changes are respected.
    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )
    base_url = os.getenv(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    )

    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=groq_api_key,
        base_url=base_url,
    )


def rag_tool(query: str) -> str:
    """Answer document-related questions using retrieval-augmented generation."""
    try:
        ranked_docs = retrieve_ranked_documents(query, k=3)
    except FileNotFoundError:
        return (
            "RAG database not found. "
            "Upload a document first at /upload-doc."
        )

    docs = [doc for doc, _ in ranked_docs]

    if not docs:
        return "I could not find relevant documents for that question."

    context = "\n\n".join(
        f"Source {i + 1}:\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs)
    )

    avg_similarity = sum(
        score for _, score in ranked_docs
    ) / len(ranked_docs)

    top_similarity = max(
        score for _, score in ranked_docs
    )

    similarity_score = (
        0.3 * avg_similarity
        + 0.7 * top_similarity
    )

    normalized_query_length = min(len(query), 120)

    try:
        confidence = predict_confidence(
            similarity_score,
            normalized_query_length,
        )
    except FileNotFoundError as exc:
        return str(exc)

    prompt = (
        "You are a helpful AI customer support assistant. "
        "Answer using the retrieved document passages. "
        "Do not invent company policies or facts. "
        "If the answer is not present in the passages, "
        "say you do not have enough information.\n\n"
        "Document passages:\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    try:
        llm = _get_chat_model()
        response = llm.invoke(prompt)
    except Exception as exc:
        error_text = str(exc).lower()

        if "model_not_found" in error_text or (
            "does not exist" in error_text and "model" in error_text
        ):
            model_name = os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            )
            return (
                f"Agent error: Groq model '{model_name}' was rejected. "
                "Check the model ID and availability in your Groq account."
            )

        if "invalid_api_key" in error_text or "401" in error_text:
            return (
                "Agent error: Groq API authentication failed. "
                "Verify GROQ_API_KEY in your .env file."
            )

        raise

    answer = (
        response.content.strip()
        if hasattr(response, "content")
        else str(response)
    )

    if confidence < 0.35:
        return (
            f"Low confidence ({confidence:.2f}). "
            "Escalating to human support.\n\n"
            f"Possible answer from retrieved documents: {answer}"
        )

    if confidence < 0.55:
        return (
            f"Tentative answer ({confidence:.2f} confidence): "
            f"{answer}"
        )

    return f"{answer}\n\n(confidence: {confidence:.2f})"


def _route_query(query: str) -> str:
    """Route the query to a support action or document retrieval."""
    lower_query = query.lower()

    # Ticket creation and complaints.
    if "ticket" in lower_query or "complaint" in lower_query:
        return create_ticket(query)

    # Refund actions.
    refund_action_words = (
        "process",
        "initiate",
        "start",
        "issue",
        "create",
    )

    has_order_reference = bool(
        re.search(
            r"\b(order|ord|rm)\s*[-:]?\s*\w+\b",
            lower_query,
        )
    )

    if "refund" in lower_query and (
        any(word in lower_query for word in refund_action_words)
        or has_order_reference
    ):
        return process_refund(query)

    # Order-status requests.
    if (
        "order status" in lower_query
        or "track order" in lower_query
        or ("order" in lower_query and "status" in lower_query)
        or "where is my order" in lower_query
    ):
        return get_order_status(query)

    # Default to RAG for policy and handbook questions.
    return rag_tool(query)


def run_agent(query: str) -> str:
    """Run the support agent and return its response."""
    try:
        return _route_query(query)

    except FileNotFoundError as exc:
        return str(exc)

    except Exception as exc:
        error_text = str(exc).lower()

        if (
            "invalid_api_key" in error_text
            or "incorrect api key" in error_text
        ):
            return (
                "Agent error: Invalid GROQ_API_KEY. "
                "Check your project-root .env file and restart the backend."
            )

        return f"Agent error: {exc}"


def stream_agent(query: str) -> Generator[str, None, None]:
    """Yield the agent response through the streaming interface."""
    yield run_agent(query)
