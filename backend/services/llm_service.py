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
import boto3  # type: ignore
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

FOLLOWUP_SYSTEM_PROMPT = """You are a helpful assistant. Your task is to suggest 
exactly 3 follow-up questions that a user might want to ask after receiving an answer.

CRITICAL REQUIREMENTS:
1. Return ONLY a JSON array of 3 strings
2. Each string MUST be a complete, grammatically correct question (ends with ?)
3. No preamble, no explanation, no markdown, no numbered lists
4. Each question should be 5-15 words long
5. Questions should be relevant to the topic but explore different angles

Example output format (and ONLY this format):
["What causes this condition?", "Are there any treatments available?", "When should I seek medical help?"]

NOW generate 3 follow-up questions based on this question and answer:"""


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
    """
    Robustly parse JSON array, extract strings, and validate they're questions.
    Returns None if parsing fails or validation fails.
    """
    # Try direct parse first
    try:
        suggestions = json.loads(text)
        if isinstance(suggestions, list) and len(suggestions) >= 3:
            validated = []
            for s in suggestions[:3]:
                s_str = str(s).strip()
                # Must be at least 5 chars and ideally end with ?
                if len(s_str) >= 5:
                    validated.append(s_str)
            if len(validated) == 3:
                return validated
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract JSON array substring
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        try:
            suggestions = json.loads(text[start : end + 1])
            if isinstance(suggestions, list) and len(suggestions) >= 3:
                validated = []
                for s in suggestions[:3]:
                    s_str = str(s).strip()
                    if len(s_str) >= 5:
                        validated.append(s_str)
                if len(validated) == 3:
                    return validated
        except json.JSONDecodeError:
            pass

    return None


def _is_valid_question(text: str) -> bool:
    """Check if text looks like a question (5+ chars, ideally ends with ?)."""
    text = text.strip()
    if len(text) < 5:
        return False
    # Ideally ends with ? but allow some wiggle
    if text.endswith(("?", ".", "!")):
        return True
    # Accept if it's phrased as a question (starts with question word)
    q_words = ("what", "why", "how", "when", "where", "who", "which", "can", "could", "should", "will", "would", "is", "are", "do", "does")
    return text.lower().startswith(q_words)


def suggest_follow_ups(query: str, answer: str) -> List[str]:
    """
    Generate 3 follow-up questions using Bedrock.
    Strict validation: returns empty list if LLM output doesn't meet requirements.
    """
    prompt = f"Question: {query}\nAnswer: {answer}"
    try:
        text, _ = _invoke(FOLLOWUP_SYSTEM_PROMPT, prompt, max_tokens=200)
    except Exception as exc:
        logger.exception("Follow-up generation failed")
        return []

    if not text.strip():
        logger.warning("Follow-up suggestions returned empty output")
        return []

    # Try strict JSON parsing first
    suggestions = _parse_json_array(text)
    if suggestions is not None:
        # Final validation: all must look like questions
        valid = [s for s in suggestions if _is_valid_question(s)]
        if len(valid) >= 2:  # Accept if at least 2 are valid
            logger.info("Follow-up suggestions validated successfully")
            return valid[:3]
        logger.warning("Follow-up JSON parsed but failed question validation")

    # If JSON parsing failed, log the raw output and return empty list
    logger.warning(
        "Follow-up suggestions failed to parse as JSON or validate as questions. "
        "Raw output: %r (first 200 chars)",
        text[:200]
    )
    return []
