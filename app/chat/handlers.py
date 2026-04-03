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
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_CJK_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_PUNCT_RE = re.compile(r"[，。！？；：、﹐﹒﹔﹕「」『』【】《》〈〉（）〔〕］［]")
_NON_JAPANESE_CJK_RE = re.compile(
    r"(\u8fd9\u662f|\u8bf7\u7528|\u65e0\u6cd5|\u6ca1\u6709)|[\u8fd9\u4eec\u4e3a\u8bf4\u5f00\u70b9\u5b9e\u89c2\u89c1\u5c06\u8ba9\u8fd8\u5417\u5e94\u6ca1]"
)
_SCOPED_FULL_CONTEXT_RETRY_MAX_CHUNKS = 12
_DATE_RANGE_RE = re.compile(
    r"((?:20\d{2}|19\d{2})年\s*\d{1,2}月\s*\d{1,2}日(?:\s*[\(（][^)）]+[\)）])?)\s*[~～\-−ー]\s*(\d{1,2}月\s*\d{1,2}日(?:\s*[\(（][^)）]+[\)）])?)"
)
_FULL_DATE_RE = re.compile(
    r"((?:20\d{2}|19\d{2})年\s*\d{1,2}月\s*\d{1,2}日(?:\s*[\(（][^)）]+[\)）])?)"
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_NUMERIC_VALUE_RE = re.compile(
    r"(?<![\d.])"
    r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"(?:\s*(万|千|百))?"
    r"(?:\s*(人|円|件|%))?"
    r"(?!\d)"
)
_NUMERIC_COMPARISON_KEYWORDS = [
    "差", "比較", "増加", "減少", "増減", "変化", "前年比", "どれくらい",
    "difference", "compare", "comparison", "increase", "decrease",
    "change", "delta", "人口", "population",
]
_NUMERIC_QUERY_STOPWORDS = {
    "資料", "文書", "ファイル", "レポート", "数値", "数字", "値", "年度", "年", "資料内",
    "記載", "ある", "で", "と", "の", "を", "は", "が",
    "差", "比較", "増加", "減少", "増減", "変化", "計算", "計算して",
    "difference", "compare", "comparison", "increase", "decrease", "change",
    "between", "from", "to", "and", "with", "value", "values", "year", "years",
    "資料", "文書", "ファイル", "数値", "数字", "値", "差", "比較", "増加", "減少", "変化",
}
_NUMERIC_METRIC_ALIASES = {
    "人口": {"人口", "population"},
    "売上": {"売上", "sales", "revenue"},
}
_CAUSAL_QUESTION_KEYWORDS = [
    "なぜ", "どうして", "理由", "原因", "要因", "背景",
    "why", "reason", "cause", "factor", "background",
]
_CAUSAL_EVIDENCE_HINTS = [
    "理由", "原因", "要因", "背景", "ため", "ために", "ことから", "によって",
    "により", "影響", "起因", "を受け", "要因として", "原因として",
    "because", "due to", "driven by", "led to", "resulted from",
]


def _needs_full_document_context(text: str) -> bool:
    """
    Detect requests that need the whole uploaded document, not just top-k chunks.

    This is used only for file-scoped chat after upload.
    """
    lower = text.lower().strip()
    indicators = [
        "要約", "まとめて", "全体", "全般", "構造", "目次",
        "分析", "分析して", "要点", "主要内容", "全体内容", "セクション", "section",
        "structure", "outline", "overview", "summarize", "summary",
        "analyze", "analysis", "key points", "main points", "extract key",
        "extract data", "important information", "important info",
        "概要", "会社概要", "企業概要", "設立", "設立日", "ビジョン", "価値",
        "核心価値", "コア価値", "主要製品", "市場の地位", "今後の展望", "結論",
    ]
    return any(keyword in lower for keyword in indicators)


def _is_smalltalk_query(text: str) -> bool:
    """Detect greetings/acknowledgements that should bypass document scoping."""
    lower = text.lower().strip()
    indicators = [
        "こんにちは", "ありがとう", "了解", "オーケー",
        "hello", "hi", "thanks", "thank you", "okay", "ok", "got it",
    ]
    return any(keyword in lower for keyword in indicators)


def _is_document_intent_query(text: str) -> bool:
    """Heuristic: whether the user is likely asking about uploaded documents."""
    lower = (text or "").lower().strip()
    indicators = [
        "文書", "ファイル", "添付", "アップロード", "pdf", "ページ", "本文", "原文",
        "根拠", "出典", "この文書", "このファイル", "その文書", "そのファイル",
        "document", "file", "attachment", "uploaded", "upload", "pdf", "page",
        "section", "paragraph", "table", "figure", "source", "quote", "lyrics",
    ]
    followup_hints = [
        "その部分", "上の内容", "さっきの内容", "それ", "そこ", "もう一度", "続けて", "続き",
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
        "その部分", "その内容", "上の内容", "さっき", "前に", "続けて", "続き", "もう一度",
        "それ", "そこ", "次", "次の条項", "次の項目", "該当内容", "その文",
        "この部分", "この内容", "この条項", "何ページ", "どのページ",
    ]
    if any(hint in lower for hint in hints):
        return True

    # Short continuation-style queries are often follow-ups.
    if len(lower) <= 40 and any(token in lower for token in ["もう一度", "続けて", "続き", "それ", "そこ", "該当"]):
        return True

    return False


def _is_scope_reset_query(text: str) -> bool:
    """Detect explicit intent to stop document-scoped conversation."""
    lower = (text or "").lower().strip()
    indicators = [
        "文書ではなく", "ファイルではなく", "アップロードではなく", "一般会話", "新しい話題", "話題を変えて",
        "別の質問", "それではなく", "文書と関係なく",
    ]
    return any(keyword in lower for keyword in indicators)


def _document_not_found_answer(
    user_message: str,
    active_source: Optional[str],
    active_doc_id: Optional[str] = None,
) -> str:
    """Return a strict grounded fallback when the document lacks evidence."""
    source_name = active_source or active_doc_id
    if source_name:
        return f"すみません。アップロードされた文書 '{source_name}' に該当する内容がないため、分かりません。"
    return "すみません。アップロードされたPDFに該当する内容がないため、分かりません。"


def _is_direct_extraction_query(text: str) -> bool:
    """Detect requests that want raw OCR/text output from the uploaded file."""
    lower = text.lower().strip()
    indicators = [
        "テキスト抽出", "文字抽出", "文字だけ", "テキストだけ", "原文", "読んで",
        "読み取って", "抽出して", "全文", "全部のテキスト", "画像の文字",
        "what does this image say", "what does the image say",
        "give me the text", "extract the text", "read the text",
        "read this image", "text in the image", "transcribe",
    ]
    return any(keyword in lower for keyword in indicators)


def _is_date_fact_query(text: str) -> bool:
    """Detect questions asking for an exact date or date range."""
    lower = (text or "").lower().strip()
    indicators = [
        "いつ", "何日", "何日から", "何日まで", "会期", "開催日", "開催期間", "日程", "期間",
        "when", "date", "dates", "period", "schedule",
    ]
    return any(keyword in lower for keyword in indicators)


def _is_causal_question(text: str) -> bool:
    """Detect questions asking for reasons, causes, or contributing factors."""
    lower = (text or "").lower().strip()
    return any(keyword in lower for keyword in _CAUSAL_QUESTION_KEYWORDS)


def _extract_causal_subject(text: str) -> Optional[str]:
    """Best-effort extraction of the phenomenon being asked about."""
    raw = re.sub(r"\s+", " ", text or "").strip()
    raw = re.sub(r"[?？!！。．]+$", "", raw)
    patterns = [
        r"^(.*?)が(?:なぜ|どうして)",
        r"^(.*?)は(?:なぜ|どうして)",
        r"^(.*?)(?:の)?(?:理由|原因|要因|背景)は",
        r"^(.*?)(?:の)?(?:理由|原因|要因|背景)を",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            subject = match.group(1).strip(" 「」『』()（）")
            if subject:
                return subject
    return None


def _has_explicit_causal_evidence(text: str) -> bool:
    """Return True when the retrieved text explicitly frames causes or factors."""
    sample = text or ""
    return any(token in sample for token in _CAUSAL_EVIDENCE_HINTS)


def _build_cautious_causal_answer(user_message: str) -> str:
    """Return a conservative answer when the document does not state a cause."""
    subject = _extract_causal_subject(user_message)
    if subject:
        if "増加" in user_message:
            return (
                f"{subject}が増加していることから、"
                "関連する要素が影響している可能性はありますが、"
                "具体的な原因については資料からは判断できません。"
            )
        elif "減少" in user_message:
            return (
                f"{subject}が減少していることから、"
                "関連する要素が影響している可能性はありますが、"
                "具体的な原因については資料からは判断できません。"
            )
        elif "変化" in user_message:
            return (
                f"{subject}が変化していることから、"
                "関連する要素が影響している可能性はありますが、"
                "具体的な原因については資料からは判断できません。"
            )
        return (
            f"{subject}に関連する記載は確認できますが、"
            "具体的な理由や要因については資料からは判断できません。"
        )

    return (
        "ご質問の事象に関連する記載は確認できますが、"
        "具体的な原因については資料からは判断できません。"
    )


def _extract_cautious_causal_answer_from_docs(user_message: str, docs) -> Optional[str]:
    """Return a safe non-causal answer when causes are not explicitly stated."""
    if not docs or not _is_causal_question(user_message):
        return None

    combined = "\n".join(
        _strip_enrichment_header(getattr(doc, "page_content", "") or "")
        for doc in docs
    )
    if not combined.strip():
        return None

    if _has_explicit_causal_evidence(combined):
        return None

    return _build_cautious_causal_answer(user_message)


def _normalize_date_text(text: str) -> str:
    cleaned = re.sub(r"\s+", "", text or "")
    cleaned = cleaned.replace("（", "（").replace(")", "）")
    return cleaned


def _extract_date_answer_from_docs(user_message: str, docs) -> Optional[str]:
    """Return an exact date/date-range answer from retrieved document text."""
    if not docs or not _is_date_fact_query(user_message):
        return None

    combined = "\n".join(
        _strip_enrichment_header(getattr(doc, "page_content", "") or "")
        for doc in docs
    )
    if not combined.strip():
        return None

    range_match = _DATE_RANGE_RE.search(combined)
    if range_match:
        start, end = range_match.groups()
        start = _normalize_date_text(start)
        end = _normalize_date_text(end)
        return f"開催期間は{start}から{end}までです。"

    date_match = _FULL_DATE_RE.search(combined)
    if date_match:
        exact_date = _normalize_date_text(date_match.group(1))
        return f"該当する日付は{exact_date}です。"

    return None


def _extract_distinct_years(text: str) -> List[str]:
    seen = set()
    years: List[str] = []
    for year in _YEAR_RE.findall(text or ""):
        if year in seen:
            continue
        seen.add(year)
        years.append(year)
    return years


def _is_numeric_comparison_query(text: str) -> bool:
    """Detect year-over-year numeric comparison questions."""
    lower = (text or "").lower().strip()
    years = _extract_distinct_years(lower)
    if len(years) < 2:
        return False
    return any(keyword in lower for keyword in _NUMERIC_COMPARISON_KEYWORDS)


def _extract_metric_tokens(text: str) -> List[str]:
    normalized = _normalize_for_match(_YEAR_RE.sub(" ", text or ""))
    tokens: List[str] = []
    seen = set()
    for token in normalized.split():
        if len(token) < 2 or token in _NUMERIC_QUERY_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens[:4]


def _resolve_metric_label(text: str) -> Optional[str]:
    normalized = _normalize_for_match(text)
    for label, aliases in _NUMERIC_METRIC_ALIASES.items():
        if any(alias.lower() in normalized for alias in aliases):
            return label
    return None


def _parse_numeric_value(raw: str, scale: Optional[str]) -> Optional[float]:
    try:
        value = float(str(raw or "").replace(",", ""))
    except ValueError:
        return None

    multiplier = {
        "万": 10_000,
        "千": 1_000,
        "百": 100,
    }.get(scale or "", 1)
    return value * multiplier


def _format_numeric_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    text = f"{value:,.2f}"
    return text.rstrip("0").rstrip(".")


def _score_numeric_candidate(
    window: str,
    line: str,
    year: str,
    metric_tokens: List[str],
    value_match: re.Match,
    parsed_value: float,
    unit: Optional[str],
    scale: Optional[str],
) -> int:
    score = 0
    normalized_window = _normalize_for_match(window)
    if metric_tokens and any(token in normalized_window for token in metric_tokens):
        score += 40
    if year in line:
        score += 20

    year_index = window.find(year)
    if year_index >= 0:
        distance = abs(value_match.start() - year_index)
        score += max(0, 30 - min(distance, 30))

    raw_match = value_match.group(0)
    if unit:
        score += 8
    if scale or "," in raw_match:
        score += 6
    if parsed_value < 100 and not unit and not scale:
        score -= 12

    tail = window[value_match.end(): value_match.end() + 1]
    if tail in {"月", "日"}:
        score -= 20

    return score


def _extract_year_value_pairs(
    text: str,
    target_years: List[str],
    metric_tokens: List[str],
) -> Dict[str, Dict[str, Any]]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    best_by_year: Dict[str, Dict[str, Any]] = {}

    for idx, line in enumerate(lines):
        window = " ".join(lines[max(0, idx - 1): min(len(lines), idx + 2)])
        for year in target_years:
            if year not in window:
                continue

            for match in _NUMERIC_VALUE_RE.finditer(window):
                raw_value, scale, unit = match.groups()
                if raw_value == year:
                    continue

                parsed_value = _parse_numeric_value(raw_value, scale)
                if parsed_value is None:
                    continue

                score = _score_numeric_candidate(
                    window=window,
                    line=line,
                    year=year,
                    metric_tokens=metric_tokens,
                    value_match=match,
                    parsed_value=parsed_value,
                    unit=unit,
                    scale=scale,
                )
                candidate = {
                    "value": parsed_value,
                    "unit": unit,
                    "score": score,
                }
                existing = best_by_year.get(year)
                if not existing or candidate["score"] > existing["score"]:
                    best_by_year[year] = candidate

    return best_by_year


def _build_numeric_comparison_answer(
    base_year: str,
    compare_year: str,
    base_value: float,
    compare_value: float,
    unit: Optional[str] = None,
    metric_label: Optional[str] = None,
) -> str:
    diff = compare_value - base_value
    unit_suffix = unit or ""
    subject = metric_label or "値"
    base_text = _format_numeric_value(base_value)
    compare_text = _format_numeric_value(compare_value)

    if unit == "%":
        point_gap = abs(diff)
        return (
            f"{base_year}年の{subject}は{base_text}%で、{compare_year}年は{compare_text}%です。"
            f" 差は{_format_numeric_value(point_gap)}ポイントです。"
        )

    if diff > 0:
        direction = "増加"
    elif diff < 0:
        direction = "減少"
    else:
        direction = "同じ"

    answer = (
        f"{base_year}年の{subject}は{base_text}{unit_suffix}で、"
        f"{compare_year}年は{compare_text}{unit_suffix}です。"
    )
    if diff == 0:
        return answer + " 差はありません。"

    answer += f" 差は{_format_numeric_value(abs(diff))}{unit_suffix}で、{direction}しています。"
    if base_value != 0:
        change_rate = abs(diff) / abs(base_value) * 100
        answer += f" 増減率は約{change_rate:.2f}%です。"
    return answer


def _extract_numeric_comparison_answer_from_docs(user_message: str, docs) -> Optional[str]:
    """Extract year-specific numeric values from retrieved docs and calculate the gap."""
    if not docs or not _is_numeric_comparison_query(user_message):
        return None

    target_years = _extract_distinct_years(user_message)
    if len(target_years) < 2:
        return None
    target_years = sorted(target_years[:2])

    combined = "\n".join(
        _strip_enrichment_header(getattr(doc, "page_content", "") or "")
        for doc in docs
    )
    if not combined.strip():
        return None

    metric_tokens = _extract_metric_tokens(user_message)
    year_values = _extract_year_value_pairs(combined, target_years, metric_tokens)
    if any(year not in year_values for year in target_years):
        return None

    base_year, compare_year = target_years
    base_entry = year_values[base_year]
    compare_entry = year_values[compare_year]
    shared_unit = base_entry.get("unit") if base_entry.get("unit") == compare_entry.get("unit") else None
    metric_label = _resolve_metric_label(user_message)

    return _build_numeric_comparison_answer(
        base_year=base_year,
        compare_year=compare_year,
        base_value=base_entry["value"],
        compare_value=compare_entry["value"],
        unit=shared_unit,
        metric_label=metric_label,
    )


def _normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^0-9a-z\uac00-\ud7a3\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff\s]", " ", text)
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


def _find_uploaded_document_by_source(source: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the newest uploaded document entry for a given source name."""
    try:
        matches = [
            doc for doc in list_documents(owner_id=owner_id, source_type="upload")
            if str(doc.get("source") or "").strip() == str(source or "").strip()
        ]
        if not matches:
            return None
        matches.sort(
            key=lambda doc: (
                str(doc.get("uploaded_at") or ""),
                str(doc.get("updated_at") or ""),
            ),
            reverse=True,
        )
        return matches[0]
    except Exception as e:
        logger.debug("Failed to resolve uploaded document by source '%s': %s", source, e)
        return None


def _get_most_recent_uploaded_document(owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the newest uploaded document for the current user."""
    try:
        docs = list_documents(owner_id=owner_id, source_type="upload")
        if not docs:
            return None
        docs.sort(
            key=lambda doc: (
                str(doc.get("uploaded_at") or ""),
                str(doc.get("updated_at") or ""),
            ),
            reverse=True,
        )
        return docs[0]
    except Exception as e:
        logger.debug("Failed to resolve most recent uploaded document: %s", e)
        return None


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
    suffix = f" ほか{len(upload_sources) - 3}件" if len(upload_sources) > 3 else ""
    return (
        "アップロードされたファイルが複数あるため、どのファイルからテキストを抽出するか分かりません。"
        f"ファイル名も一緒に指定してください。 (例: {preview}{suffix})"
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


def _docs_have_machine_readable_text(docs) -> bool:
    """Return True when retrieved docs came from a real PDF text layer."""
    if not docs:
        return False

    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        if meta.get("has_text_layer") is True:
            return True
        if meta.get("extraction_method") == "text":
            return True
        if meta.get("page_kind") == "digital":
            return True

    return False


def _format_direct_extraction_text(user_message: str, extracted: str) -> str:
    """Return raw extracted text with a localized label and no interpretation."""
    return f"抽出されたテキスト:\n\n{extracted}"


def _document_read_failure_answer(user_message: str, active_source: Optional[str] = None) -> str:
    """Return a localized refusal when OCR/text extraction looks unreliable."""
    source_name = active_source or "アップロードされた文書"
    return (
        f"すみません。'{source_name}' は正確に読み取れませんでした。"
        "内容が分かりません。"
    )


def _looks_unreliable_extracted_text(text: str, user_message: str = "") -> bool:
    """Heuristic: detect OCR text that is likely noise and should not be interpreted."""
    sample = (text or "").strip()
    if not sample:
        return True

    if _looks_garbled_output(sample):
        return True

    meaningful_chars = len(re.findall(r"[\uac00-\ud7a3A-Za-z0-9\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", sample))
    if meaningful_chars < 12:
        return True

    latin_tokens = re.findall(r"\b[A-Za-z0-9]{4,}\b", sample)
    uppercase_tokens = [tok for tok in latin_tokens if tok.upper() == tok and re.search(r"[A-Z]", tok)]
    alpha_tokens = re.findall(r"\b[A-Za-z]{4,}\b", sample)
    upperish_tokens = [
        tok for tok in alpha_tokens
        if (sum(1 for ch in tok if ch.isupper()) / max(1, len(tok))) >= 0.6
    ]
    uppercase_count = len(re.findall(r"[A-Z]", sample))
    ascii_alpha_count = len(re.findall(r"[A-Za-z]", sample))
    lowercase_count = len(re.findall(r"[a-z]", sample))
    japanese_count = len(re.findall(r"[\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", sample))
    uppercase_ratio = (uppercase_count / ascii_alpha_count) if ascii_alpha_count else 0.0

    if latin_tokens and len(uppercase_tokens) >= 4 and len(uppercase_tokens) >= int(len(latin_tokens) * 0.7):
        if lowercase_count == 0 and japanese_count == 0:
            return True

    if _JAPANESE_RE.search(user_message):
        if japanese_count == 0 and len(uppercase_tokens) >= 3 and lowercase_count == 0:
            return True
        if japanese_count <= 2 and ascii_alpha_count >= 40 and uppercase_ratio >= 0.55 and len(upperish_tokens) >= 5:
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

    if (not _docs_have_machine_readable_text(docs)) and _looks_unreliable_extracted_text(extracted, user_message=user_message):
        fallback_source = active_source or (docs[0].metadata.get("source") if docs else None)
        return _document_read_failure_answer(user_message, fallback_source)

    return _format_direct_extraction_text(user_message, extracted)


def _retry_scoped_retrieval_with_full_document(
    user_message: str,
    scoped_source: Optional[str],
    scoped_doc_id: Optional[str],
    scoped_source_type: Optional[str],
    owner_id: Optional[str] = None,
) -> Optional[Any]:
    """
    When scoped retrieval is too weak, retry once with the full document.

    This prevents immediate "not found" fallbacks for recent uploads where
    semantic retrieval misses broad questions like "summarize this PDF".
    """
    if not (scoped_source or scoped_doc_id):
        return None

    try:
        chunk_count = get_document_chunk_count(
            source=scoped_source,
            doc_id=scoped_doc_id,
            source_type=scoped_source_type,
            owner_id=owner_id,
        )
    except Exception as e:
        logger.debug("Could not inspect scoped document for full-context retry: %s", e)
        return None

    if chunk_count <= 0 or chunk_count > _SCOPED_FULL_CONTEXT_RETRY_MAX_CHUNKS:
        return None

    retry = retrieve(
        user_message,
        source_filter=scoped_source,
        doc_id_filter=scoped_doc_id,
        source_type_filter=scoped_source_type,
        owner_id_filter=owner_id,
        full_document=True,
    )
    if retry.docs:
        logger.info(
            "Retrying scoped query with full document context: %d chunks from '%s'%s",
            len(retry.docs),
            scoped_source or "scoped document",
            f" ({scoped_doc_id})" if scoped_doc_id else "",
        )
        return retry

    return None


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
1. Respond ONLY in natural Japanese.
2. Never output non-Japanese text.
3. Never output non-Japanese CJK phrasing. Japanese kanji are allowed only as part of natural Japanese sentences.
4. Answer ONLY from the retrieved document context. Do not use general knowledge, prior assumptions, or web knowledge.
5. If the retrieved document context is missing, weak, unrelated, or insufficient, say you do not know in Japanese.
6. Never guess, summarize from memory, or fill gaps with plausible information.
7. If the user asks for a reason, cause, factor, or background, do not infer causality unless the document explicitly states it.
8. If the document shows a result or change but does not explicitly state its cause, say that the cause cannot be determined from the document.
9. Do not mention source names, file names, page numbers, citations, references, or evidence labels unless the user explicitly asks for them.
10. Keep paragraphs clean. Do not produce broken line fragments or punctuation-only lines.
11. Do NOT use markdown emphasis symbols in the final answer (forbidden: **, __). Output plain text only."""


def _build_prompt(
    user_message: str,
    history: List[Message],
    doc_context: str = "",
    system_prompt: str = "",
) -> str:
    """Build a single unified prompt with all available context."""
    parts = [f"[System]\n{_SYSTEM_PROMPT}"]

    extra_system_prompt = (system_prompt or "").strip()
    if extra_system_prompt:
        parts.append(
            "[Additional user preferences - lower priority than System rules]\n"
            + extra_system_prompt
        )

    history_text = _format_history(history)
    if history_text:
        parts.append(f"[Conversation history]\n{history_text}")

    if doc_context:
        parts.append(f"[Retrieved document context]\n{doc_context}")

    parts.append(f"[User message]\n{user_message}")

    return "\n\n".join(parts)


def _contains_han_chars(text: str) -> bool:
    """Return True when any CJK Han character exists."""
    return bool(_CJK_HAN_RE.search(text or ""))


def _contains_cjk_punctuation(text: str) -> bool:
    """Return True when CJK full-width punctuation appears."""
    return bool(_CJK_PUNCT_RE.search(text or ""))


def _normalize_cjk_punctuation(text: str) -> str:
    """Normalize CJK punctuation into ASCII punctuation for cleaner output."""
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
        meaningful = len(re.findall(r"[\uac00-\ud7a3A-Za-z0-9\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", ln))
        has_cjk_punct = bool(_CJK_PUNCT_RE.search(ln))
        if has_cjk_punct and meaningful <= 1:
            noisy_lines += 1

        # e.g., "- 。：，。" style broken list lines
        if re.fullmatch(r"[-•\s，。！？；：,.;:!?]+", ln) and len(ln) >= 2:
            noisy_lines += 1

    return noisy_lines >= 2


def _merge_wrapped_lines(previous: str, current: str) -> str:
    """Join accidental soft-wraps without damaging Japanese text."""
    if not previous:
        return current
    if re.search(r"[A-Za-z0-9]$", previous) and re.search(r"^[A-Za-z0-9]", current):
        return previous + " " + current
    return previous + current


def _normalize_answer_layout(text: str) -> str:
    """Collapse broken line-wraps and punctuation-only fragments."""
    if not text:
        return text

    raw_lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]

    normalized: List[str] = []
    for line in raw_lines:
        if not line:
            if normalized and normalized[-1] != "":
                normalized.append("")
            continue

        if re.fullmatch(r"[，。、！？；：,.;:!?・\-•\s]+", line):
            continue

        if not normalized or normalized[-1] == "" or re.match(r"^([\-*•]|[0-9]+[.)]|[・■□])\s+", line):
            normalized.append(line)
            continue

        if re.search(r"[。．.!?！？]$", normalized[-1]):
            normalized.append(line)
            continue

        normalized[-1] = _merge_wrapped_lines(normalized[-1], line)

    cleaned = "\n".join(normalized).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _strip_markdown_emphasis(text: str) -> str:
    """Remove markdown emphasis/heading markers from model output."""
    if not text:
        return text
    cleaned = text.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    return cleaned


def _strip_source_reference_lines(text: str) -> str:
    """Remove source/citation style lines from final answers."""
    if not text:
        return text

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(参考|参考資料|参考ドキュメント|出典|根拠|参照)\s*[:：]", stripped):
            continue
        if re.match(r"^(source|sources|reference|references|citation|citations)\s*[:：]", stripped, flags=re.IGNORECASE):
            continue
        if re.match(r"^(document|page)\s*[:：]", stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\(\s*(?:p|page)\.?\s*\d+\s*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:page|p)\.?\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _contains_hangul(user_message: str) -> bool:
    """Detect whether the user message contains Hangul."""
    return bool(_HANGUL_RE.search(user_message or ""))


def _expects_japanese_output(user_message: str) -> bool:
    """Treat messages containing kana as Japanese-mode conversations."""
    return bool(_JAPANESE_RE.search(user_message or ""))


def _expected_output_language(user_message: str) -> str:
    """This app is locked to Japanese output."""
    return "ja"


def _looks_non_japanese_cjk_output(text: str) -> bool:
    """Detect CJK-heavy output that does not read like natural Japanese."""
    sample = _normalize_answer_layout(text or "")
    meaningful = len(re.findall(r"[A-Za-z0-9\uac00-\ud7a3\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", sample))
    kana_count = len(_JAPANESE_RE.findall(sample))
    han_count = len(_CJK_HAN_RE.findall(sample))
    hangul_count = len(_HANGUL_RE.findall(sample))
    latin_count = len(re.findall(r"[A-Za-z]", sample))

    if hangul_count > 0:
        return True

    if _NON_JAPANESE_CJK_RE.search(sample):
        return True

    if meaningful >= 8 and han_count >= 3 and kana_count == 0:
        return True

    if meaningful >= 16 and han_count >= 6 and kana_count < max(2, han_count // 8):
        return True

    if meaningful >= 16 and kana_count == 0 and latin_count >= 6:
        return True

    return False


def _is_acceptable_japanese_output(text: str) -> bool:
    """Require readable Japanese with kana present in non-trivial answers."""
    sample = _normalize_answer_layout(_strip_markdown_emphasis(text or ""))
    if not sample:
        return False
    if _NON_JAPANESE_CJK_RE.search(sample):
        return False
    if _looks_garbled_output(sample):
        return False
    if _looks_non_japanese_cjk_output(sample):
        return False

    meaningful = len(re.findall(r"[A-Za-z0-9\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", sample))
    kana_count = len(_JAPANESE_RE.findall(sample))
    han_count = len(_CJK_HAN_RE.findall(sample))

    if meaningful >= 14 and kana_count == 0:
        return False
    if meaningful >= 18 and han_count >= 8 and kana_count < 2:
        return False

    return True


def _needs_language_rewrite(user_message: str, answer: str) -> bool:
    """
    Rewrite when:
    1) non-Japanese CJK markers exist, or
    2) output is not acceptable Japanese, or
    3) Output looks garbled/noisy.
    """
    if not answer:
        return False

    expected_lang = _expected_output_language(user_message)

    if _looks_garbled_output(answer):
        return True

    if expected_lang == "ja":
        if _contains_han_chars(answer) and _looks_non_japanese_cjk_output(answer):
            return True
        if _contains_cjk_punctuation(answer) and _looks_garbled_output(answer):
            return True
        if not _is_acceptable_japanese_output(answer):
            return True

    return False


def _rewrite_answer_to_user_language(answer: str, user_message: str, model: str) -> str:
    """One-shot rewrite pass to force natural Japanese and clean layout."""
    rewrite_prompt = f"""[System]
You are a strict editor.
Respond ONLY in natural Japanese.
Every non-trivial sentence must read like native Japanese and should use hiragana/katakana where appropriate.
ABSOLUTE RULES:
- Do not output non-Japanese text.
- If non-Japanese CJK text exists in the draft, translate/paraphrase it into natural Japanese.
- Remove source names, file names, page numbers, citation labels, and reference sections unless the user explicitly asked for them.
- Remove broken line wraps and punctuation-only fragments.
Preserve facts, order, and citations from the draft answer.
Do not add new claims.

[User message]
{user_message}

[Draft answer]
{answer}

[Task]
Rewrite the draft into clean natural Japanese only.
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
Output only Japanese.
Do not output non-Japanese text.
Remove broken line wraps.
Preserve meaning and important details.

[Draft answer]
{answer}
""".strip()
    rewritten = call_ollama(prompt, model=model, temperature=0.0)
    return get_response_text(rewritten)


def _apply_language_guard(user_message: str, answer: str, model: str) -> str:
    """
    Hard guard:
    - Never allow non-Japanese CJK output.
    - Enforce stable Japanese output.
    """
    expected_lang = _expected_output_language(user_message)
    expect_japanese = expected_lang == "ja"

    if not _needs_language_rewrite(user_message, answer):
        return _normalize_answer_layout(answer)

    candidate = answer
    try:
        for _ in range(3):
            rewritten = _rewrite_answer_to_user_language(candidate, user_message, model)
            if not rewritten:
                break
            candidate = rewritten
            candidate = _normalize_answer_layout(candidate)

            cjk_is_clean = not _looks_non_japanese_cjk_output(candidate)
            no_cjk_punct = not _looks_garbled_output(candidate)
            has_japanese = _is_acceptable_japanese_output(candidate)
            not_garbled = not _looks_garbled_output(candidate)
            if (
                cjk_is_clean
                and no_cjk_punct
                and not_garbled
                and (not expect_japanese or has_japanese)
            ):
                logger.info("Applied strict language guard.")
                return _strip_markdown_emphasis(candidate).strip()
    except Exception as e:
        logger.warning("Strict language guard rewrite failed: %s", e)

    # Normalize any remaining non-Japanese CJK noise first.
    stripped = _normalize_answer_layout(candidate or "")
    stripped = re.sub(r"[ 	]{2,}", " ", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()

    # If still garbled, ask for one more clean rewrite in user language.
    if _looks_garbled_output(stripped):
        try:
            cleaned = _rewrite_answer_to_user_language(stripped, user_message, model)
            if cleaned:
                cleaned = _normalize_answer_layout(cleaned)
                cleaned = _strip_markdown_emphasis(cleaned).strip()
                if cleaned and _is_acceptable_japanese_output(cleaned):
                    logger.warning("Applied garbled-output cleanup rewrite in language guard.")
                    return cleaned
        except Exception as e:
            logger.warning("Garbled-output cleanup rewrite failed: %s", e)

    # If Japanese output is still not acceptable, force one translation pass.
    if expect_japanese and not _is_acceptable_japanese_output(stripped):
        try:
            translated = _translate_to_target_language_fallback(stripped or candidate, model, "Japanese")
            translated = _normalize_answer_layout(translated)
            if translated and _is_acceptable_japanese_output(translated):
                translated = _strip_markdown_emphasis(translated).strip()
                logger.warning("Applied Japanese fallback translation in language guard.")
                return translated
        except Exception as e:
            logger.warning("Japanese fallback translation failed: %s", e)

        return "日本語で回答するよう再試行しましたが、変換に失敗しました。もう一度同じ質問をしてください。"

    if stripped and _is_acceptable_japanese_output(stripped):
        logger.warning("Applied hard-strip fallback in language guard.")
        return stripped

    return "日本語の応答を安定して生成できませんでした。同じ質問をもう一度試してください。"


# ═══════════════════════════════════════════════════════════════════════

def _has_token_repetition_loop(answer: str) -> bool:
    """
    Detect degenerate repetitive answers like:
    "student student student ..."
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
            inferred_doc = _find_uploaded_document_by_source(inferred_source, owner_id=user_id)
            if inferred_doc:
                scoped_doc_id = scoped_doc_id or inferred_doc.get("doc_id")

    if not (scoped_source or scoped_doc_id) and document_intent:
        recent_upload = _get_most_recent_uploaded_document(owner_id=user_id)
        if recent_upload:
            scoped_source = str(recent_upload.get("source") or "").strip() or None
            scoped_doc_id = str(recent_upload.get("doc_id") or "").strip() or None
            scoped_source_type = scoped_source_type or "upload"
            logger.info(
                "Recovered upload scope from most recent document: '%s'%s",
                scoped_source or "unknown",
                f" ({scoped_doc_id})" if scoped_doc_id else "",
            )

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
                if (not _docs_have_machine_readable_text(docs)) and _looks_unreliable_extracted_text(fresh_text, user_message=user_message):
                    extraction_answer = _document_read_failure_answer(user_message, scoped_source or active_source)
                else:
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
            or _is_numeric_comparison_query(user_message)
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
                retry = _retry_scoped_retrieval_with_full_document(
                    user_message,
                    scoped_source,
                    scoped_doc_id,
                    scoped_source_type,
                    owner_id=user_id,
                )
                if retry:
                    retrieval = retry
                    docs = retrieval.docs
                    use_full_document = True
                else:
                    return {
                        "answer": _document_not_found_answer(
                            user_message,
                            scoped_source or active_source or ("uploaded files" if scoped_source_type == "upload" else None),
                            scoped_doc_id or active_doc_id,
                        ),
                        "sources": [],
                        "mode": "document_qa",
                        "active_source": scoped_source or active_source,
                        "active_doc_id": scoped_doc_id or active_doc_id,
                    }

            return {
                "answer": _document_not_found_answer(
                    user_message,
                    scoped_source or active_source or ("uploaded PDFs" if scoped_source_type == "upload" else None),
                    scoped_doc_id or active_doc_id,
                ),
                "sources": [],
                "mode": "document_qa",
                "active_source": scoped_source or active_source,
                "active_doc_id": scoped_doc_id or active_doc_id,
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
        if (
            has_any_scope
            and combined_scoped_text
            and (not _docs_have_machine_readable_text(docs))
            and _looks_unreliable_extracted_text(combined_scoped_text, user_message=user_message)
        ):
            return {
                "answer": _document_read_failure_answer(
                    user_message,
                    scoped_source or active_source or active_doc_id,
                ),
                "sources": sources,
                "mode": "document_qa",
                "active_source": scoped_source or active_source,
                "active_doc_id": scoped_doc_id or active_doc_id,
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
            "active_source": scoped_source or active_source,
            "active_doc_id": scoped_doc_id or active_doc_id,
        }

    exact_date_answer = _extract_date_answer_from_docs(user_message, docs)
    if exact_date_answer:
        resolved_active_source = scoped_source or active_source
        resolved_active_doc_id = scoped_doc_id or active_doc_id
        if (not resolved_active_source or not resolved_active_doc_id) and sources:
            first_source = sources[0] or {}
            resolved_active_source = resolved_active_source or first_source.get("source")
            resolved_active_doc_id = resolved_active_doc_id or first_source.get("doc_id")
        return {
            "answer": exact_date_answer,
            "sources": sources,
            "mode": "document_qa",
            "active_source": resolved_active_source,
            "active_doc_id": resolved_active_doc_id,
        }

    numeric_comparison_answer = _extract_numeric_comparison_answer_from_docs(user_message, docs)
    if numeric_comparison_answer:
        resolved_active_source = scoped_source or active_source
        resolved_active_doc_id = scoped_doc_id or active_doc_id
        if (not resolved_active_source or not resolved_active_doc_id) and sources:
            first_source = sources[0] or {}
            resolved_active_source = resolved_active_source or first_source.get("source")
            resolved_active_doc_id = resolved_active_doc_id or first_source.get("doc_id")
        return {
            "answer": numeric_comparison_answer,
            "sources": sources,
            "mode": "document_qa",
            "active_source": resolved_active_source,
            "active_doc_id": resolved_active_doc_id,
        }

    cautious_causal_answer = _extract_cautious_causal_answer_from_docs(user_message, docs)
    if cautious_causal_answer:
        resolved_active_source = scoped_source or active_source
        resolved_active_doc_id = scoped_doc_id or active_doc_id
        if (not resolved_active_source or not resolved_active_doc_id) and sources:
            first_source = sources[0] or {}
            resolved_active_source = resolved_active_source or first_source.get("source")
            resolved_active_doc_id = resolved_active_doc_id or first_source.get("doc_id")
        return {
            "answer": cautious_causal_answer,
            "sources": sources,
            "mode": "document_qa",
            "active_source": resolved_active_source,
            "active_doc_id": resolved_active_doc_id,
        }

    # ── Step 2: Build prompt with retrieved document evidence only ──
    effective_system_prompt = (system_prompt or "").strip()
    if _is_causal_question(user_message):
        causal_guard = (
            "If the user asks for a reason, cause, factor, or background, "
            "do not infer causality unless it is explicitly stated in the retrieved document context. "
            "If the cause is not explicit, say that it cannot be determined from the document."
        )
        effective_system_prompt = (
            f"{effective_system_prompt}\n\n{causal_guard}".strip()
            if effective_system_prompt
            else causal_guard
        )

    prompt = _build_prompt(
        user_message=user_message,
        history=history,
        doc_context=doc_context,
        system_prompt=effective_system_prompt,
    )

    result = call_ollama(prompt, model=selected_model)
    answer = get_response_text(result)
    answer = _apply_language_guard(user_message, answer, selected_model)
    answer = _apply_repetition_guard(user_message, answer, selected_model)
    answer = _strip_markdown_emphasis(answer)
    answer = _strip_source_reference_lines(answer)
    answer = _normalize_answer_layout(answer)

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
