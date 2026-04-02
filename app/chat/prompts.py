"""
Prompt templates for each chat mode.

FIX: All prompts now match the user's language instead of hardcoding Korean.
The LLM (qwen2.5) is Chinese-origin and defaults to Chinese when confused.
Explicit language matching + "NEVER respond in Chinese" prevents this.
"""

from typing import List
from app.models.schemas import Message


def format_history(history: List[Message], max_turns: int = 8) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-max_turns:]:
        lines.append(f"[{msg.role}]\n{msg.content}")
    return "\n\n".join(lines)


# Language instruction that goes into every prompt
_LANG_RULE = (
    "重要: ユーザーが使用している言語と同じ言語で応答してください。 "
    "ユーザーが日本語で書いている場合は、日本語で応答してください。 "
    "ユーザーが英語で書いている場合は、英語で応答してください。 "
    "いかなる状況においても中国語(中文)では応答しないでください。"
)


# ═══════════════════════════════════════════════════════════════════════
# GENERAL CHAT
# ═══════════════════════════════════════════════════════════════════════

def build_general_prompt(
    system_prompt: str,
    history: List[Message],
    user_message: str,
) -> str:
    return f"""[システム指針]
{system_prompt}

回答ルール:
1. {_LANG_RULE}
2. ユーザーの入力が短い挨拶や単純な表現の場合は、簡潔で自然な応答をしてください。
3. 前の文脈を無理に繋げないでください。
4. わからないことは「わかりません」と答えてください。

[対話履歴]
{format_history(history)}

[ユーザーの質問]
{user_message}""".strip()


# ═══════════════════════════════════════════════════════════════════════
# DOCUMENT QA  ← This is what the fine-tuned model will receive
# ═══════════════════════════════════════════════════════════════════════

def build_document_prompt(
    system_prompt: str,
    history: List[Message],
    user_message: str,
    context: str,
) -> str:
    """
    Context format from retriever.py:format_context():
    [Doc: filename.pdf | Page: 3 | Section: title | Lang: ja]
    <chunk text here>
    """
    return f"""[システム指針]
{system_prompt}

回答ルール:
1. {_LANG_RULE}
2. 提供されたドキュメントコンテキストのみに基づいて回答してください。
3. ドキュメントにない内容は推測しないでください。
4. コンテキストが不足している場合は「その内容は提供されたドキュメントに記載されていません」と答えてください。
5. まずコア回答を述べてください。
6. 出典、ページ番号、参照ドキュメント名は、ユーザーが明示的に求めた場合を除き書かないでください。
7. 画像から抽出されたテキストが提供される場合は、そのテキストに基づいて答えてください。

[対話履歴]
{format_history(history)}

[検索されたドキュメント]
{context if context else "関連するドキュメントが見つかりません"}

[ユーザーの質問]
{user_message}

[回答形式]
- コア回答:
- 要点:""".strip()


# ═══════════════════════════════════════════════════════════════════════
# WEB SEARCH
# ═══════════════════════════════════════════════════════════════════════

def build_web_prompt(
    system_prompt: str,
    history: List[Message],
    user_message: str,
    search_results: str = "",
) -> str:
    search_section = ""
    if search_results:
        search_section = f"\n[ウェブ検索結果]\n{search_results}\n"

    return f"""[システム指針]
{system_prompt}

回答ルール:
1. {_LANG_RULE}
2. これは最新情報が必要な質問です。
3. ウェブ検索結果が提供される場合は、その情報を優先的に参照してください。
4. 最新情報が不確かな場合は、不確かであることを述べてください。
5. 要点を簡潔かつ明確にまとめてください。
{search_section}
[対話履歴]
{format_history(history)}

[ユーザーの質問]
{user_message}""".strip()
