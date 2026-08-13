"""
Direct Bedrock calls, replacing the old LangChain RetrievalQA chain +
local llama-2-7b-chat.ggmlv3 CTransformers model. Building the prompt by
hand here makes it straightforward to control token budget, chat history,
and to add the "general chat" and "follow-up questions" behaviors described
in the reconstructed architecture.
"""
import json
import time
from typing import Optional, List, Tuple
import boto3
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_client = None

RAG_SYSTEM_PROMPT = """You are Medibot, a careful medical-information assistant.
Answer the user's question using ONLY the provided context excerpts.
If the answer is not contained in the context, say you don't know rather
than guessing. Keep answers concise and clear. Do not provide a diagnosis;
frame information as general medical knowledge and suggest consulting a
healthcare professional for personal medical decisions."""

GENERAL_CHAT_SYSTEM_PROMPT = """You are Medibot, a friendly assistant for a
medical-document Q&A tool. The user has sent a greeting or a general
question unrelated to any uploaded document. Respond briefly and warmly,
and if relevant, invite them to ask a question about their uploaded
documents."""

FOLLOWUP_SYSTEM_PROMPT = """Given the user's question and the answer that
was provided, suggest exactly 3 short, relevant follow-up questions the
user might want to ask next. Respond ONLY as a JSON array of 3 strings,
nothing else."""


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            region_name=settings.aws_region,
        )
    return _client


def _collect_text_content(content) -> List[str]:
    texts = []
    if isinstance(content, str):
        return [content]

    if isinstance(content, dict):
        for key, value in content.items():
            if key in {"text", "output_text"} and isinstance(value, str):
                texts.append(value)
            elif key == "content" and isinstance(value, str):
                texts.append(value)
            elif isinstance(value, (dict, list)):
                texts.extend(_collect_text_content(value))
        return texts

    if isinstance(content, list):
        for item in content:
            texts.extend(_collect_text_content(item))
        return texts

    return texts


def _extract_text_from_response(payload: dict) -> str:
    """Extract text from either Bedrock or OpenAI-style response formats."""
    # Try Bedrock format first
    text = "".join(_collect_text_content(payload.get("content", [])))
    if text:
        return text
    
    # Try OpenAI format
    if "choices" in payload and isinstance(payload["choices"], list):
        for choice in payload["choices"]:
            if isinstance(choice, dict):
                # Try message.content path (OpenAI)
                if "message" in choice and isinstance(choice["message"], dict):
                    content = choice["message"].get("content", "")
                    if content:
                        return content
                # Try text path (some APIs)
                if "text" in choice:
                    text_val = choice.get("text", "")
                    if text_val:
                        return text_val
    
    # Fallback to output_text or text at root level
    return payload.get("output_text", "") or payload.get("text", "")


def _invoke(system_prompt: str, user_prompt: str, max_tokens: int = 512) -> Tuple[str, float]:
    client = _get_client()
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    start = time.perf_counter()
    try:
        response = client.invoke_model(
            modelId=settings.bedrock_model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
    except Exception as exc:
        logger.exception("Bedrock model invocation failed")
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000

    try:
        payload = json.loads(response["body"].read())
    except Exception as exc:
        logger.exception("Failed to parse Bedrock response JSON")
        raise

    text = _extract_text_from_response(payload)
    if not text:
        logger.warning(
            "Model response contained no text; payload keys=%s",
            list(payload.keys()),
        )
    return text.strip(), elapsed_ms


def is_general_chat(query: str) -> bool:
    """Cheap heuristic to route greetings/small talk away from retrieval."""
    normalized = query.strip().lower()
    greetings = {"hi", "hello", "hey", "thanks", "thank you", "who are you", "what can you do"}
    return normalized in greetings or len(normalized.split()) <= 2 and any(
        g in normalized for g in ["hi", "hello", "hey", "thanks"]
    )


def answer_general_chat(query: str) -> Tuple[str, float]:
    return _invoke(GENERAL_CHAT_SYSTEM_PROMPT, query, max_tokens=200)


def answer_with_context(query: str, context: str) -> Tuple[str, float]:
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    return _invoke(RAG_SYSTEM_PROMPT, user_prompt, max_tokens=512)


def _parse_json_array(text: str) -> Optional[List[str]]:
    try:
        suggestions = json.loads(text)
        if isinstance(suggestions, list):
            return [str(s) for s in suggestions if isinstance(s, (str, int, float))]
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract a JSON array substring from model output.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        try:
            suggestions = json.loads(text[start : end + 1])
            if isinstance(suggestions, list):
                return [str(s) for s in suggestions if isinstance(s, (str, int, float))]
        except json.JSONDecodeError:
            pass

    return None


def _parse_follow_up_text(text: str) -> List[str]:
    lines = [line.strip().lstrip("- ").lstrip("0123456789. ") for line in text.splitlines() if line.strip()]
    suggestions = [line for line in lines if len(line) > 3]
    return suggestions[:3]


def suggest_follow_ups(query: str, answer: str) -> List[str]:
    prompt = f"Question: {query}\nAnswer: {answer}"
    text, _ = _invoke(FOLLOWUP_SYSTEM_PROMPT, prompt, max_tokens=200)
    if not text.strip():
        logger.warning("Follow-up suggestions returned empty output")
        return []

    suggestions = _parse_json_array(text)
    if suggestions is not None:
        return suggestions[:3]

    suggestions = _parse_follow_up_text(text)
    if suggestions:
        logger.warning("Follow-up suggestions returned non-JSON text, using fallback parse.")
        return suggestions

    logger.warning("Failed to parse follow-up suggestions: raw output=%r", text)
    return []
