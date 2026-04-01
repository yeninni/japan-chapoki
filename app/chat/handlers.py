"""
Unified chat handler for document-grounded chat.

Behavior:
1. Always search the vectorstore for relevant document context
2. Answer only from retrieved document evidence
3. If the evidence is missing or unreliable, say you do not know
"""

import logging
from collections import Counter
from pathlib import Path
import re
from typing import List, Dict, Any, Optional

from app.models.schemas import Message
from app.core.llm import call_ollama, get_response_text
from app.retrieval.retriever import retrieve, format_context, extract_sources
from app.core.vectorstore import get_documents_by_source, get_document_chunk_count
from app.core.document_registry import list_documents
from app.pipeline.parser import extract_full_text
from app.config import (
    OLLAMA_MODEL,
    DOCUMENT_CONFIDENCE_THRESHOLD,
    SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD,
    SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS,
    UPLOADS_DIR,
)

logger = logging.getLogger("tilon.chat")
_HANGUL_RE = re.compile(r"[가-힣]")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_CJK_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_PUNCT_RE = re.compile(r"[，。！？；：、﹐﹒﹔﹕「」『』【】《》〈〉（）〔〕］［]")


def _needs_full_document_context(text: str) -> bool:
    """
    Detect requests that need the whole uploaded document, not just top-k chunks.

    This is used only for file-scoped chat after upload.
    """
    lower = text.lower().strip()
    indicators = [
        "요약", "요약해", "정리", "정리해", "전체", "전반", "구조", "목차",
        "분석", "분석해", "핵심", "주요 내용", "전체 내용", "섹션", "section",
        "structure", "outline", "overview", "summarize", "summary",
        "analyze", "analysis", "key points", "main points", "extract key",
        "extract data", "important information", "important info",
    ]
    return any(keyword in lower for keyword in indicators)


def _is_smalltalk_query(text: str) -> bool:
    """Detect greetings/acknowledgements that should bypass document scoping."""
    lower = text.lower().strip()
    indicators = [
        "안녕", "고마워", "감사", "오케이", "알겠", "응", "네",
        "hello", "hi", "thanks", "thank you", "okay", "ok", "got it",
    ]
    return any(keyword in lower for keyword in indicators)


def _is_document_intent_query(text: str) -> bool:
    """Heuristic: whether the user is likely asking about uploaded documents."""
    lower = (text or "").lower().strip()
    indicators = [
        "문서", "파일", "첨부", "업로드", "pdf", "페이지", "쪽", "본문", "원문",
        "가사", "근거", "출처", "해당 문서", "이 문서", "이 파일", "그 문서", "그 파일",
        "document", "file", "attachment", "uploaded", "upload", "pdf", "page",
        "section", "paragraph", "table", "figure", "source", "quote", "lyrics",
    ]
    followup_hints = [
        "그 부분", "위 내용", "방금 내용", "그거", "거기", "다시", "이어", "계속",
    ]

    if any(keyword in lower for keyword in indicators):
        return True

    # Very short follow-up utterances are often document continuation questions.
    if len(lower) <= 20 and any(hint in lower for hint in followup_hints):
        return True

    return False


def _is_scope_followup_query(text: str) -> bool:
    """Heuristic: follow-up utterance likely referring to previously scoped document chat."""
    lower = (text or "").lower().strip()
    if not lower:
        return False

    hints = [
        "그 부분", "그 내용", "위 내용", "방금", "앞에서", "이어서", "계속", "다시",
        "그거", "거기", "그 다음", "다음 조항", "다음 항목", "해당 내용", "그 문장",
        "이 부분", "이 내용", "이 조항", "몇 페이지", "어느 페이지",
    ]
    if any(hint in lower for hint in hints):
        return True

    # Short continuation-style queries are often follow-ups.
    if len(lower) <= 40 and any(token in lower for token in ["다시", "이어", "계속", "그거", "거기", "해당"]):
        return True

    return False


def _is_scope_reset_query(text: str) -> bool:
    """Detect explicit intent to stop document-scoped conversation."""
    lower = (text or "").lower().strip()
    indicators = [
        "문서 말고", "파일 말고", "업로드 말고", "일반 대화", "새 주제", "주제 바꿔",
        "이제 다른 질문", "그거 말고", "문서와 상관없이",
    ]
    return any(keyword in lower for keyword in indicators)


def _document_not_found_answer(
    user_message: str,
    active_source: Optional[str],
    active_doc_id: Optional[str] = None,
) -> str:
    """Return a strict grounded fallback when the document lacks evidence."""
    source_name = active_source or active_doc_id
    if _HANGUL_RE.search(user_message):
        if source_name:
            return f"죄송합니다. 업로드된 문서 '{source_name}'에서 해당 내용을 찾지 못해 알 수 없습니다."
        return "죄송합니다. 업로드된 PDF 문서에서 해당 내용을 찾지 못해 알 수 없습니다."
    if _JAPANESE_RE.search(user_message):
        if source_name:
            return f"すみません。アップロードされた文書 '{source_name}' に該当する内容がないため、分かりません。"
        return "すみません。アップロードされたPDFに該当する内容がないため、分かりません。"
    if source_name:
        return f"Sorry, I don't know because the uploaded document '{source_name}' does not contain that information."
    return "Sorry, I don't know because the uploaded PDF does not contain that information."


def _is_direct_extraction_query(text: str) -> bool:
    """Detect requests that want raw OCR/text output from the uploaded file."""
    lower = text.lower().strip()
    indicators = [
        "텍스트 추출", "문자 추출", "글자 추출", "읽어줘", "텍스트만", "원문", "ocr",
        "모든 텍스트", "전체 텍스트", "텍스트 다", "다 뽑", "전문",
        "テキスト抽出", "文字抽出", "文字だけ", "テキストだけ", "原文", "読んで",
        "読み取って", "抽出して", "全文", "全部のテキスト", "画像の文字",
        "what does this image say", "what does the image say",
        "give me the text", "extract the text", "read the text",
        "read this image", "text in the image", "transcribe",
    ]
    return any(keyword in lower for keyword in indicators)


def _normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^0-9a-z가-힣\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _list_uploaded_sources(owner_id: Optional[str] = None) -> List[str]:
    """Return unique uploaded source names from lightweight document registry."""
    try:
        seen = set()
        sources = []
        for doc in list_documents(owner_id=owner_id, source_type="upload"):
            if not doc or doc.get("source_type") != "upload":
                continue
            source = str(doc.get("source") or "").strip()
            if not source:
                continue
            key = source.lower()
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
        return sorted(sources)
    except Exception as e:
        logger.debug("Failed to list uploaded sources: %s", e)
        return []


def _infer_upload_source_from_query_or_history(user_message: str, history: List[Message], owner_id: Optional[str] = None) -> Optional[str]:
    """Best-effort: infer which uploaded file the user is referring to."""
    sources = _list_uploaded_sources(owner_id=owner_id)
    if not sources:
        return None

    msg_norm = _normalize_for_match(user_message)
    user_hist = [m.content for m in (history or []) if getattr(m, "role", "") == "user"]
    recent_hist = user_hist[-8:]

    best_source = None
    best_score = 0

    for source in sources:
        stem = Path(source).stem
        stem_norm = _normalize_for_match(stem)
        if not stem_norm:
            continue

        score = 0
        # Strong signal: full stem appears in current query.
        if stem_norm in msg_norm:
            score += 120

        # Token overlap signal.
        tokens = [tok for tok in stem_norm.split(" ") if len(tok) >= 2]
        token_hits = sum(1 for tok in tokens if tok in msg_norm)
        if token_hits >= 2:
            score += 50
        elif token_hits == 1 and any(len(tok) >= 4 and tok in msg_norm for tok in tokens):
            score += 20

        # Conversation carry-over: recently mentioned file in user turns.
        for idx, text in enumerate(reversed(recent_hist), start=1):
            hist_norm = _normalize_for_match(text)
            if stem_norm and stem_norm in hist_norm:
                score += max(0, 40 - (idx * 5))
                break

        if score > best_score:
            best_source = source
            best_score = score

    if best_score >= 35:
        logger.info("Inferred upload source '%s' (score=%d) from query/history", best_source, best_score)
        return best_source

    return None


def _direct_extraction_ambiguous_answer(upload_sources: List[str], user_message: str = "") -> str:
    preview = ", ".join(upload_sources[:3])
    suffix = f" 외 {len(upload_sources) - 3}개" if len(upload_sources) > 3 else ""
    if _JAPANESE_RE.search(user_message):
        return (
            "アップロードされたファイルが複数あるため、どのファイルからテキストを抽出するか分かりません。"
            f"ファイル名も一緒に指定してください。 (例: {preview}{suffix})"
        )
    return (
        "업로드 파일이 여러 개라 어떤 파일에서 텍스트를 뽑아야 할지 애매합니다. "
        f"파일명을 같이 말해 주세요. (예: {preview}{suffix})"
    )


def _extract_text_from_upload_source(source: Optional[str], user_id: Optional[str] = None, source_path: Optional[str] = None) -> str:
    """Extract fresh text from uploaded source file to avoid stale/noisy chunk artifacts."""
    if not source:
        return ""
    try:
        candidate_paths: List[Path] = []

        if source_path:
            candidate_paths.append(Path(source_path))

        safe_name = Path(source).name
        if user_id:
            candidate_paths.append(UPLOADS_DIR / Path(user_id).name / safe_name)
        candidate_paths.append(UPLOADS_DIR / safe_name)

        for candidate in candidate_paths:
            if candidate.exists() and candidate.is_file():
                return (extract_full_text(str(candidate)) or "").strip()
        return ""
    except Exception as e:
        logger.debug("Direct source extraction failed for %s: %s", source, e)
        return ""


def _strip_enrichment_header(text: str) -> str:
    """Remove enrichment header prepended before embedding/retrieval."""
    return re.sub(r'^\[Document:.*?\]\n', '', text or '', flags=re.DOTALL).strip()


def _format_direct_extraction_text(user_message: str, extracted: str) -> str:
    """Return raw extracted text with a localized label and no interpretation."""
    if _HANGUL_RE.search(user_message):
        return f"추출된 텍스트:\n\n{extracted}"
    if _JAPANESE_RE.search(user_message):
        return f"抽出されたテキスト:\n\n{extracted}"
    if _HANGUL_RE.search(extracted):
        return f"추출된 텍스트:\n\n{extracted}"
    return f"Extracted text:\n\n{extracted}"


def _document_read_failure_answer(user_message: str, active_source: Optional[str] = None) -> str:
    """Return a localized refusal when OCR/text extraction looks unreliable."""
    source_name = active_source or "the uploaded document"
    if _JAPANESE_RE.search(user_message):
        return (
            f"すみません。'{source_name}' は正確に読み取れませんでした。"
            "内容が分かりません。"
        )
    if _HANGUL_RE.search(user_message):
        return (
            f"죄송합니다. '{source_name}' 문서를 정확하게 읽지 못했습니다. "
            "내용을 알 수 없습니다."
        )
    return (
        f"Sorry, I couldn't read '{source_name}' accurately. "
        "I don't know the content."
    )


def _looks_unreliable_extracted_text(text: str, user_message: str = "") -> bool:
    """Heuristic: detect OCR text that is likely noise and should not be interpreted."""
    sample = (text or "").strip()
    if not sample:
        return True

    if _looks_garbled_output(sample):
        return True

    meaningful_chars = len(re.findall(r"[가-힣A-Za-z0-9\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", sample))
    if meaningful_chars < 12:
        return True

    latin_tokens = re.findall(r"\b[A-Za-z0-9]{4,}\b", sample)
    uppercase_tokens = [tok for tok in latin_tokens if tok.upper() == tok and re.search(r"[A-Z]", tok)]
    lowercase_count = len(re.findall(r"[a-z]", sample))
    japanese_count = len(re.findall(r"[\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", sample))

    if latin_tokens and len(uppercase_tokens) >= 4 and len(uppercase_tokens) >= int(len(latin_tokens) * 0.7):
        if lowercase_count == 0 and japanese_count == 0:
            return True

    if _JAPANESE_RE.search(user_message):
        if japanese_count == 0 and len(uppercase_tokens) >= 3 and lowercase_count == 0:
            return True

    return False


def _build_direct_extraction_answer(active_source: Optional[str], docs, user_message: str = "") -> str:
    """Return extracted document text directly for OCR/transcription requests."""
    extracted = "\n\n".join(
        _strip_enrichment_header(doc.page_content)
        for doc in docs
        if _strip_enrichment_header(doc.page_content)
    ).strip()

    if not extracted:
        fallback_source = active_source or (docs[0].metadata.get("source") if docs else None)
        fallback_doc_id = docs[0].metadata.get("doc_id") if docs else None
        return _document_not_found_answer(user_message or "extract text", fallback_source, fallback_doc_id)

    if _looks_unreliable_extracted_text(extracted, user_message=user_message):
        fallback_source = active_source or (docs[0].metadata.get("source") if docs else None)
        return _document_read_failure_answer(user_message, fallback_source)

    return _format_direct_extraction_text(user_message, extracted)


def _scoped_confidence_threshold(docs) -> float:
    """
    Use a slightly lower threshold for tiny uploaded docs where one chunk is
    effectively the whole document (e.g. a screenshot or one-page upload).
    """
    if len(docs) != 1:
        return DOCUMENT_CONFIDENCE_THRESHOLD

    meta = docs[0].metadata
    if meta.get("source_type") == "upload":
        return SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD

    return DOCUMENT_CONFIDENCE_THRESHOLD


def _should_force_small_doc_full_context(
    active_source: Optional[str],
    active_doc_id: Optional[str],
    owner_id: Optional[str] = None,
) -> bool:
    """Use full-document context for tiny scoped docs where top-k retrieval is brittle."""
    if not active_source and not active_doc_id:
        return False

    try:
        chunk_count = get_document_chunk_count(source=active_source, doc_id=active_doc_id, owner_id=owner_id)
    except Exception as e:
        logger.debug("Could not inspect scoped document chunk count: %s", e)
        return False

    if chunk_count <= 0:
        return False

    return chunk_count <= SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS


# ═══════════════════════════════════════════════════════════════════════
# Prompt Builder
# ═══════════════════════════════════════════════════════════════════════

def _format_history(history: List[Message], max_turns: int = 8) -> str:
    if not history:
        return ""
    return "\n\n".join(
        f"[{msg.role}]\n{msg.content}" for msg in history[-max_turns:]
    )


_SYSTEM_PROMPT = """You are Tilon AI, a strict document-based chatbot.

CRITICAL RULES:
1. Respond in the SAME language the user is using. Korean → Korean. English → English. NEVER output Chinese characters (中文/汉字).
2. Answer ONLY from the retrieved document context. Do not use general knowledge, prior assumptions, or web knowledge.
3. If the retrieved document context is missing, weak, unrelated, or insufficient, say you do not know.
4. Never guess, summarize from memory, or fill gaps with plausible information.
5. When document evidence is available, cite the source naturally (document name, page number).
6. Be concise and direct. Answer the question first, then explain if needed.
7. If retrieved text includes Chinese characters, translate/paraphrase them into the user's language instead of copying Chinese characters.
8. Do NOT use markdown emphasis symbols in the final answer (forbidden: **, __). Output plain text only."""


def _build_prompt(
    user_message: str,
    history: List[Message],
    doc_context: str = "",
    system_prompt: str = "",
) -> str:
    """Build a single unified prompt with all available context."""
    parts = [f"[System]\n{system_prompt or _SYSTEM_PROMPT}"]

    history_text = _format_history(history)
    if history_text:
        parts.append(f"[Conversation history]\n{history_text}")

    if doc_context:
        parts.append(f"[Retrieved document context]\n{doc_context}")

    parts.append(f"[User message]\n{user_message}")

    return "\n\n".join(parts)


def _contains_chinese_chars(text: str) -> bool:
    """Return True when any CJK Han character exists."""
    return bool(_CJK_HAN_RE.search(text or ""))


def _contains_cjk_punctuation(text: str) -> bool:
    """Return True when CJK full-width punctuation appears."""
    return bool(_CJK_PUNCT_RE.search(text or ""))


def _normalize_cjk_punctuation(text: str) -> str:
    """Normalize CJK punctuation into ASCII punctuation for clean Korean output."""
    if not text:
        return text
    table = str.maketrans({
        "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
        "﹐": ",", "﹒": ".", "﹔": ";", "﹕": ":",
        "（": "(", "）": ")", "［": "[", "］": "]",
    })
    normalized = text.translate(table)
    normalized = re.sub(r"[「」『』【】《》〈〉〔〕]", "", normalized)
    return normalized


def _looks_garbled_output(answer: str) -> bool:
    """Detect noisy/degenerated outputs (punctuation fragments, broken lines)."""
    text = (answer or "").strip()
    if not text:
        return False

    if re.search(r"[，。！？；：]\s*[，。！？；：]", text):
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    noisy_lines = 0
    for ln in lines:
        # Lines with mostly punctuation and almost no meaningful letters/numbers.
        meaningful = len(re.findall(r"[가-힣A-Za-z0-9\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", ln))
        has_cjk_punct = bool(_CJK_PUNCT_RE.search(ln))
        if has_cjk_punct and meaningful <= 1:
            noisy_lines += 1

        # e.g., "- 。：，。" style broken list lines
        if re.fullmatch(r"[-•\s，。！？；：,.;:!?]+", ln) and len(ln) >= 2:
            noisy_lines += 1

    return noisy_lines >= 2


def _strip_markdown_emphasis(text: str) -> str:
    """Remove markdown emphasis/heading markers from model output."""
    if not text:
        return text
    cleaned = text.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    return cleaned


def _expects_korean_output(user_message: str) -> bool:
    """Treat messages containing Hangul as Korean-mode conversations."""
    return bool(_HANGUL_RE.search(user_message or ""))


def _expects_japanese_output(user_message: str) -> bool:
    """Treat messages containing kana as Japanese-mode conversations."""
    return bool(_JAPANESE_RE.search(user_message or ""))


def _expected_output_language(user_message: str) -> str:
    """Infer the primary response language from the user message."""
    if _expects_korean_output(user_message):
        return "ko"
    if _expects_japanese_output(user_message):
        return "ja"
    return "other"


def _needs_language_rewrite(user_message: str, answer: str) -> bool:
    """
    Rewrite when:
    1) Chinese chars/punctuation exist, or
    2) User wrote in Korean but answer is effectively non-Korean, or
    3) Output looks garbled/noisy.
    """
    if not answer:
        return False

    expected_lang = _expected_output_language(user_message)

    if (expected_lang != "ja" and _contains_chinese_chars(answer)) or (
        expected_lang != "ja" and _contains_cjk_punctuation(answer)
    ):
        return True

    if _looks_garbled_output(answer):
        return True

    if expected_lang == "ko":
        hangul_count = len(_HANGUL_RE.findall(answer))
        latin_count = len(re.findall(r"[A-Za-z]", answer))
        # If there is no Hangul and enough Latin text, force Korean rewrite.
        if hangul_count == 0 and latin_count >= 5:
            return True
    elif expected_lang == "ja":
        if _HANGUL_RE.search(answer):
            return True
        kana_count = len(_JAPANESE_RE.findall(answer))
        latin_count = len(re.findall(r"[A-Za-z]", answer))
        han_count = len(_CJK_HAN_RE.findall(answer))
        if kana_count == 0 and han_count == 0 and latin_count >= 5:
            return True

    return False


def _rewrite_answer_to_user_language(answer: str, user_message: str, model: str) -> str:
    """One-shot rewrite pass to remove Chinese and align language with the user."""
    rewrite_prompt = f"""[System]
You are a strict editor.
Respond in the SAME language as the user message.
If user message is Korean, output must include natural Korean sentences (Hangul).
If user message is Japanese, output must include natural Japanese sentences.
ABSOLUTE RULE: Do not output Chinese characters (中文/汉字) at all.
If Chinese text exists in the draft, translate/paraphrase it into the user language.
Preserve facts, order, and citations from the draft answer.
Do not add new claims.

[User message]
{user_message}

[Draft answer]
{answer}

[Task]
Rewrite the draft to match the user language and remove Chinese characters.
""".strip()

    rewritten = call_ollama(
        rewrite_prompt,
        model=model,
        temperature=0.0,
    )
    return get_response_text(rewritten)


def _translate_to_target_language_fallback(answer: str, model: str, target_language: str) -> str:
    """Last-resort translation into the user's language."""
    prompt = f"""[System]
Translate the draft answer into natural {target_language}.
Do not output Chinese characters (中文/汉字).
Preserve meaning and important details.

[Draft answer]
{answer}
""".strip()
    rewritten = call_ollama(prompt, model=model, temperature=0.0)
    return get_response_text(rewritten)


def _apply_language_guard(user_message: str, answer: str, model: str) -> str:
    """
    Hard guard:
    - Never allow Chinese characters.
    - Enforce the user's language when it is clearly detectable.
    """
    expected_lang = _expected_output_language(user_message)
    expect_korean = expected_lang == "ko"
    expect_japanese = expected_lang == "ja"

    if not _needs_language_rewrite(user_message, answer):
        return answer

    candidate = answer
    try:
        for _ in range(3):
            rewritten = _rewrite_answer_to_user_language(candidate, user_message, model)
            if not rewritten:
                break
            candidate = rewritten

            no_chinese = expect_japanese or not _contains_chinese_chars(candidate)
            no_cjk_punct = expect_japanese or not _contains_cjk_punctuation(candidate)
            has_korean = bool(_HANGUL_RE.search(candidate))
            has_japanese = bool(_JAPANESE_RE.search(candidate))
            not_garbled = not _looks_garbled_output(candidate)
            if (
                no_chinese
                and no_cjk_punct
                and not_garbled
                and (not expect_korean or has_korean)
                and (not expect_japanese or has_japanese)
            ):
                logger.info("Applied strict language guard.")
                return _strip_markdown_emphasis(candidate).strip()
    except Exception as e:
        logger.warning("Strict language guard rewrite failed: %s", e)

    # Remove any remaining Chinese chars first and normalize punctuation noise.
    stripped = candidate or ""
    if not expect_japanese:
        stripped = _CJK_HAN_RE.sub("", stripped)
    if not expect_japanese:
        stripped = _normalize_cjk_punctuation(stripped)
    stripped = re.sub(r"[ 	]{2,}", " ", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()

    # If still garbled, ask for one more clean rewrite in user language.
    if _looks_garbled_output(stripped):
        try:
            cleaned = _rewrite_answer_to_user_language(stripped, user_message, model)
            if cleaned:
                if not expect_japanese:
                    cleaned = _normalize_cjk_punctuation(cleaned)
                cleaned = _strip_markdown_emphasis(cleaned).strip()
                if cleaned and (expect_japanese or not _contains_chinese_chars(cleaned)) and not _looks_garbled_output(cleaned):
                    logger.warning("Applied garbled-output cleanup rewrite in language guard.")
                    return cleaned
        except Exception as e:
            logger.warning("Garbled-output cleanup rewrite failed: %s", e)

    # If user language is expected but still absent, force one translation pass.
    if expect_korean and not _HANGUL_RE.search(stripped):
        try:
            translated = _translate_to_target_language_fallback(stripped or candidate, model, "Korean")
            if translated:
                translated = _normalize_cjk_punctuation(translated)
            if translated and _HANGUL_RE.search(translated) and not _contains_chinese_chars(translated):
                translated = _strip_markdown_emphasis(translated).strip()
                if not _contains_cjk_punctuation(translated) and not _looks_garbled_output(translated):
                    logger.warning("Applied Korean fallback translation in language guard.")
                    return translated
        except Exception as e:
            logger.warning("Korean fallback translation failed: %s", e)

        return "한국어로 답변하도록 재시도했지만 변환에 실패했습니다. 같은 질문을 다시 입력해 주세요."

    if expect_japanese and not _JAPANESE_RE.search(stripped):
        try:
            translated = _translate_to_target_language_fallback(stripped or candidate, model, "Japanese")
            if translated and _JAPANESE_RE.search(translated):
                translated = _strip_markdown_emphasis(translated).strip()
                if not _looks_garbled_output(translated):
                    logger.warning("Applied Japanese fallback translation in language guard.")
                    return translated
        except Exception as e:
            logger.warning("Japanese fallback translation failed: %s", e)

        return "日本語で回答するよう再試行しましたが、変換に失敗しました。もう一度同じ質問をしてください。"

    if stripped:
        logger.warning("Applied hard-strip fallback in language guard.")
        return stripped

    if expect_japanese:
        return "言語ポリシーに合う応答を生成できませんでした。同じ質問をもう一度試してください。"
    return "언어 정책에 맞는 응답 생성에 실패했습니다. 같은 질문을 다시 시도해 주세요."


# ═══════════════════════════════════════════════════════════════════════

def _has_token_repetition_loop(answer: str) -> bool:
    """
    Detect degenerate repetitive answers like:
    "대학원생 대학원생 대학원생 ..."
    """
    tokens = [t for t in re.split(r"\s+", (answer or "").strip()) if t]
    if len(tokens) < 20:
        return False

    max_run = 1
    current_run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    if max_run >= 8:
        return True

    dominant_token, dominant_count = Counter(tokens).most_common(1)[0]
    if len(dominant_token) >= 2 and dominant_count >= max(12, int(len(tokens) * 0.35)):
        return True

    return False


def _rewrite_repetitive_answer(answer: str, user_message: str, model: str) -> str:
    """Rewrite one repetitive answer into a concise non-redundant form."""
    rewrite_prompt = f"""[System]
You are a careful editor.
Respond in the SAME language as the user message.
Remove repetitive loops and redundant phrases.
Do not add new facts.
Preserve important points and source mentions from the draft.

[User message]
{user_message}

[Draft answer]
{answer}

[Task]
Rewrite the draft so it is concise, readable, and non-repetitive.
""".strip()

    rewritten = call_ollama(
        rewrite_prompt,
        model=model,
        temperature=0.0,
    )
    return get_response_text(rewritten)


def _apply_repetition_guard(user_message: str, answer: str, model: str) -> str:
    """Post-process repetitive outputs with one rewrite pass."""
    if not _has_token_repetition_loop(answer):
        return answer

    try:
        rewritten = _rewrite_repetitive_answer(answer, user_message, model)
        rewritten = _apply_language_guard(user_message, rewritten, model)
        if rewritten:
            logger.info("Applied repetition guard (rewrote degenerate repetitive answer).")
            return rewritten
    except Exception as e:
        logger.warning("Repetition guard rewrite failed: %s", e)

    return answer
# Unified Chat Handler
# ═══════════════════════════════════════════════════════════════════════

def handle_chat(
    user_message: str,
    history: List[Message] = None,
    model: str = None,
    active_source: str = None,
    active_doc_id: str = None,
    active_source_type: str = None,
    system_prompt: str = None,
    web_search_enabled: bool = False,
    user_id: str = None,
) -> Dict[str, Any]:
    """
    Unified chat handler — works like a normal chatbot.

    Args:
        user_message: What the user said
        history: Conversation history
        model: Which Ollama model to use
        active_source: Active document filename scope for the current chat, if any
        active_doc_id: Stable document ID scope for the current chat, if any
        active_source_type: Source-type scope for the current chat (e.g., 'upload'), if any
        system_prompt: Override default system prompt
        web_search_enabled: Reserved for backward compatibility; web search is disabled
    """
    history = history or []
    selected_model = model or OLLAMA_MODEL
    scoped_source = None if _is_smalltalk_query(user_message) else active_source
    scoped_doc_id = None if _is_smalltalk_query(user_message) else active_doc_id
    scoped_source_type = None if _is_smalltalk_query(user_message) else active_source_type
    has_initial_scope = bool(scoped_source or scoped_doc_id or scoped_source_type)

    scope_reset_requested = _is_scope_reset_query(user_message)
    document_intent = (
        _is_document_intent_query(user_message)
        or _is_direct_extraction_query(user_message)
        or _needs_full_document_context(user_message)
        or (has_initial_scope and _is_scope_followup_query(user_message))
        or (has_initial_scope and not scope_reset_requested)
    )

    if scope_reset_requested:
        document_intent = False

    # Keep explicit chat-level document scope unless the user clearly resets it.
    if not document_intent:
        scoped_source = None
        scoped_doc_id = None
        scoped_source_type = None

    if not (scoped_source or scoped_doc_id) and scoped_source_type == "upload":
        inferred_source = _infer_upload_source_from_query_or_history(user_message, history, owner_id=user_id)
        if inferred_source:
            scoped_source = inferred_source

    if _is_direct_extraction_query(user_message) and not (scoped_source or scoped_doc_id) and scoped_source_type == "upload":
        upload_sources = _list_uploaded_sources(owner_id=user_id)
        return {
            "answer": _direct_extraction_ambiguous_answer(upload_sources, user_message=user_message),
            "sources": [],
            "mode": "ocr_extract",
            "active_source": active_source,
            "active_doc_id": active_doc_id,
        }

    if (scoped_source or scoped_doc_id) and _is_direct_extraction_query(user_message):
        docs = get_documents_by_source(source=scoped_source, doc_id=scoped_doc_id, owner_id=user_id)
        if docs:
            logger.info(
                "Direct extraction response for '%s'%s",
                scoped_source or "scoped document",
                f" ({scoped_doc_id})" if scoped_doc_id else "",
            )

            fresh_text = _extract_text_from_upload_source(scoped_source, user_id=user_id, source_path=(docs[0].metadata.get("source_path") if docs else None))
            if fresh_text:
                extraction_answer = _format_direct_extraction_text(user_message, fresh_text)
            else:
                extraction_answer = _build_direct_extraction_answer(scoped_source, docs, user_message=user_message)

            return {
                "answer": extraction_answer,
                "sources": extract_sources(docs),
                "mode": "ocr_extract",
                "active_source": scoped_source,
                "active_doc_id": scoped_doc_id,
            }

    # ── Step 1: Always search for relevant document context ──
    doc_context = ""
    sources = []

    use_full_document = bool(
        (scoped_source or scoped_doc_id)
        and (
            _needs_full_document_context(user_message)
            or _should_force_small_doc_full_context(scoped_source, scoped_doc_id)
        )
    )
    retrieval = retrieve(
        user_message,
        source_filter=scoped_source,
        doc_id_filter=scoped_doc_id,
        source_type_filter=scoped_source_type,
        owner_id_filter=user_id,
        full_document=use_full_document,
    )
    docs = retrieval.docs

    has_explicit_scope = bool(scoped_source or scoped_doc_id)
    has_source_type_scope = bool(scoped_source_type)
    has_any_scope = has_explicit_scope or has_source_type_scope

    if has_any_scope and not use_full_document:
        threshold = _scoped_confidence_threshold(docs) if docs else DOCUMENT_CONFIDENCE_THRESHOLD
        low_confidence = (not docs) or (
            retrieval.confidence < threshold
            and not retrieval.strong_keyword_hit
        )
        if low_confidence:
            logger.info(
                "Low-confidence scoped retrieval for '%s'%s (confidence=%.2f, threshold=%.2f, keyword_hit=%s)",
                active_source or active_source_type or "scoped document",
                f" ({scoped_doc_id})" if scoped_doc_id else "",
                retrieval.confidence,
                threshold,
                retrieval.strong_keyword_hit,
            )

            # Explicit per-file scope should remain strict.
            if has_explicit_scope:
                return {
                    "answer": _document_not_found_answer(
                        user_message,
                        active_source or ("uploaded files" if active_source_type == "upload" else None),
                        active_doc_id,
                    ),
                    "sources": [],
                    "mode": "document_qa",
                    "active_source": active_source,
                    "active_doc_id": active_doc_id,
                }

            return {
                "answer": _document_not_found_answer(
                    user_message,
                    active_source or ("uploaded PDFs" if active_source_type == "upload" else None),
                    active_doc_id,
                ),
                "sources": [],
                "mode": "document_qa",
                "active_source": active_source,
                "active_doc_id": active_doc_id,
            }

    if docs and not has_any_scope and not use_full_document:
        low_confidence = (
            retrieval.confidence < DOCUMENT_CONFIDENCE_THRESHOLD
            and not retrieval.strong_keyword_hit
        )
        if low_confidence:
            logger.info(
                "Low-confidence unscoped retrieval (confidence=%.2f, threshold=%.2f, keyword_hit=%s)",
                retrieval.confidence,
                DOCUMENT_CONFIDENCE_THRESHOLD,
                retrieval.strong_keyword_hit,
            )
            return {
                "answer": _document_not_found_answer(user_message, None, None),
                "sources": [],
                "mode": "document_qa",
                "active_source": active_source,
                "active_doc_id": active_doc_id,
            }

    if docs:
        doc_context = format_context(docs)
        sources = extract_sources(docs)

        combined_scoped_text = "\n\n".join(
            _strip_enrichment_header(doc.page_content)
            for doc in docs
            if _strip_enrichment_header(doc.page_content)
        ).strip()
        if has_any_scope and combined_scoped_text and _looks_unreliable_extracted_text(combined_scoped_text, user_message=user_message):
            return {
                "answer": _document_read_failure_answer(
                    user_message,
                    scoped_source or active_source or active_doc_id,
                ),
                "sources": sources,
                "mode": "document_qa",
                "active_source": active_source,
                "active_doc_id": active_doc_id,
            }

        if use_full_document:
            logger.info(
                "Loaded full document context: %d chunks from '%s'",
                len(docs),
                scoped_source,
            )
        else:
            logger.info(
                "Found %d relevant chunks%s (confidence=%.2f, keyword_hit=%s)",
                len(docs),
                (
                    " (scoped to "
                    + ", ".join(
                        part
                        for part in [
                            f"source='{scoped_source}'" if scoped_source else "",
                            f"doc_id='{scoped_doc_id}'" if scoped_doc_id else "",
                            f"source_type='{scoped_source_type}'" if scoped_source_type else "",
                        ]
                        if part
                    )
                    + ")"
                ) if (scoped_source or scoped_doc_id or scoped_source_type) else "",
                retrieval.confidence,
                retrieval.strong_keyword_hit,
            )
    else:
        logger.info("No relevant document chunks found")
        return {
            "answer": _document_not_found_answer(
                user_message,
                scoped_source or active_source or ("uploaded PDFs" if scoped_source_type == "upload" else None),
                scoped_doc_id or active_doc_id,
            ),
            "sources": [],
            "mode": "document_qa",
            "active_source": active_source,
            "active_doc_id": active_doc_id,
        }

    # ── Step 2: Build prompt with retrieved document evidence only ──
    prompt = _build_prompt(
        user_message=user_message,
        history=history,
        doc_context=doc_context,
        system_prompt=system_prompt,
    )

    result = call_ollama(prompt, model=selected_model)
    answer = get_response_text(result)
    answer = _apply_language_guard(user_message, answer, selected_model)
    answer = _apply_repetition_guard(user_message, answer, selected_model)
    answer = _strip_markdown_emphasis(answer)

    # Determine what was used (for UI display)
    mode = "document_qa"

    resolved_active_source = None
    resolved_active_doc_id = None
    if document_intent:
        resolved_active_source = scoped_source or active_source
        resolved_active_doc_id = scoped_doc_id or active_doc_id
        if (not resolved_active_source or not resolved_active_doc_id) and sources:
            first_source = sources[0] or {}
            resolved_active_source = resolved_active_source or first_source.get("source")
            resolved_active_doc_id = resolved_active_doc_id or first_source.get("doc_id")

    return {
        "answer": answer,
        "sources": sources,
        "mode": mode,
        "active_source": resolved_active_source,
        "active_doc_id": resolved_active_doc_id,
    }
