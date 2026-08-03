"""MCP server for you-know — expose the knowledge model to any AI via MCP.

v0.2 — New tools:
  - extract_terms: Auto-extract technical terms from text (passive growth)
  - analyze_conversation: Analyze a conversation turn for passive inference
  - get_learning_summary: Learning dashboard (Pull mode entry point)

Run: python -m you_know.server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .engine import (
    analyze_conversation,
    check_answer,
    get_learning_summary,
)
from .matcher import auto_register_terms, extract_technical_terms, ConceptMatcher
from .store import KnowledgeStore

# ── Globals ────────────────────────────────────────────────
_store: KnowledgeStore | None = None
_matcher: ConceptMatcher | None = None
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store


def get_matcher() -> ConceptMatcher:
    global _matcher
    if _matcher is None:
        _matcher = ConceptMatcher(get_store())
    return _matcher


# ── Tool implementations ───────────────────────────────────

async def tool_lookup_concept(term: str) -> str:
    """Look up a concept by id, name, or alias."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    concept = store.lookup(term)
    if concept is None:
        return json.dumps({
            "found": False,
            "term": term,
            "suggestion": "Concept not in graph. Use add_concept to register it.",
        }, ensure_ascii=False)

    return json.dumps({
        "found": True,
        "id": concept.id,
        "name": concept.name,
        "explanation": concept.explanation,
        "status": concept.status.value,
        "confidence": round(concept.confidence, 3),
        "depth_level": concept.depth_level.value,
        "stability": round(concept.stability, 3),
        "aliases": concept.aliases,
        "parent_ids": concept.parent_ids,
        "needs_explanation": concept.needs_explanation(),
    }, ensure_ascii=False)


async def tool_check_answer(text: str, max_depth: int | None = None) -> str:
    """Check an answer text. Returns unknown concepts and optionally expands."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    matcher = get_matcher()
    result = check_answer(text, store, max_depth=max_depth, matcher=matcher)
    return json.dumps(result, ensure_ascii=False, indent=2)


async def tool_extract_terms(text: str) -> str:
    """Extract technical terms from text and auto-register unknown ones.

    This is the passive growth engine — each call grows the concept graph.
    Safe to call on every AI response (lightweight, no LLM involvement).
    """
    store = get_store()
    if not store.graph.concepts:
        store.load()

    matcher = get_matcher()

    # First, show what terms were found
    candidates = extract_technical_terms(text)
    new_ids = auto_register_terms(text, store)
    if new_ids:
        matcher.mark_dirty()
        store.save()

    return json.dumps({
        "candidates_found": len(candidates),
        "candidates": candidates[:20],  # Top 20
        "newly_registered": len(new_ids),
        "new_concept_ids": new_ids,
        "total_concepts": len(store.graph.concepts),
    }, ensure_ascii=False, indent=2)


async def tool_analyze_conversation(user_message: str, ai_response: str = "") -> str:
    """Analyze a conversation turn for passive concept inference.

    Detects:
      - Concepts the user demonstrates understanding of
      - Concepts the user is confused about
      - Auto-registers new terms from AI response

    Call this AFTER each user message (not on the critical path).
    """
    store = get_store()
    if not store.graph.concepts:
        store.load()

    matcher = get_matcher()
    result = analyze_conversation(user_message, ai_response, store, matcher)

    # Save if anything changed
    if result["inferences"] or result["new_concepts_registered"]:
        store.save()

    return json.dumps(result, ensure_ascii=False, indent=2)


async def tool_get_learning_summary() -> str:
    """Get a learning dashboard summary.

    The Pull mode entry point — user calls this to see:
      - Knowledge stats (total/known/learning/unknown)
      - Strongest concepts
      - Concepts needing attention
      - Blind spots (never engaged)
      - Forgetting curve decay alerts
    """
    store = get_store()
    if not store.graph.concepts:
        store.load()

    summary = get_learning_summary(store)
    return json.dumps(summary, ensure_ascii=False, indent=2)


async def tool_mark_known(term: str, evidence: str = "") -> str:
    """Mark a concept as understood by the user."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    concept = store.mark_known(term, evidence)
    if concept is None:
        concept = store.upsert_concept(
            id=term.lower().replace(" ", "-"),
            name=term,
            explanation=f"User understands {term} (confirmed directly)",
            status="known",
        )
        concept = store.mark_known(concept.id, evidence)

    return json.dumps({
        "ok": True,
        "id": concept.id,
        "name": concept.name,
        "status": concept.status.value,
        "confidence": round(concept.confidence, 3),
    }, ensure_ascii=False)


async def tool_mark_learning(term: str, evidence: str = "") -> str:
    """Mark a concept as currently being learned."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    concept = store.graph.mark_learning(term, evidence)
    if concept is None:
        concept = store.upsert_concept(
            id=term.lower().replace(" ", "-"),
            name=term,
            explanation=f"User is learning about {term}",
            status="learning",
        )
        concept = store.graph.mark_learning(concept.id, evidence)

    return json.dumps({
        "ok": True,
        "id": concept.id,
        "name": concept.name,
        "status": concept.status.value,
        "confidence": round(concept.confidence, 3),
    }, ensure_ascii=False)


async def tool_add_concept(
    id: str, name: str, explanation: str,
    aliases: str = "", parent_ids: str = "",
    status: str = "unknown",
    depth_level: str = "exposed",
) -> str:
    """Add or update a concept in the knowledge graph."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    alias_list = [a.strip() for a in aliases.split(",") if a.strip()] if aliases else []
    parent_list = [p.strip() for p in parent_ids.split(",") if p.strip()] if parent_ids else []

    concept = store.upsert_concept(
        id=id, name=name, explanation=explanation,
        aliases=alias_list, parent_ids=parent_list,
        status=status,
    )

    return json.dumps({
        "ok": True,
        "id": concept.id,
        "name": concept.name,
        "status": concept.status.value,
    }, ensure_ascii=False)


async def tool_get_stats() -> str:
    """Get knowledge graph statistics with depth breakdown."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    stats = store.graph.stats()
    return json.dumps(stats, ensure_ascii=False)


async def tool_list_concepts(status: str = "") -> str:
    """List concepts, optionally filtered by status."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    if status:
        concepts = [c for c in store.graph.concepts.values()
                    if c.status.value == status]
    else:
        concepts = list(store.graph.concepts.values())

    return json.dumps(
        [{"id": c.id, "name": c.name, "status": c.status.value,
          "confidence": round(c.confidence, 3), "depth": c.depth_level.value}
         for c in concepts],
        ensure_ascii=False, indent=2,
    )


# ── Server setup ───────────────────────────────────────────

def _tool(name, desc, schema, required=None):
    """Helper to create Tool objects with less boilerplate."""
    return Tool(
        name=name,
        description=desc,
        inputSchema={
            "type": "object",
            "properties": schema,
            "required": required or [],
        },
    )


TOOLS = [
    _tool("lookup_concept", "查询一个概念：用户是否理解它？返回完整状态包括认知深度和置信度。",
          {"term": {"type": "string", "description": "概念标识符（id、名称或别名）"}},
          ["term"]),

    _tool("check_answer",
          "【核心工具】分析一段回答文本，找出用户不理解的术语，广度优先展开解释（默认 depth=2）。"
          "同时自动注册文本中出现的新技术术语。",
          {"text": {"type": "string", "description": "要检查的回答文本"},
           "max_depth": {"type": "integer", "description": "最大展开深度（默认自适应计算）"}},
          ["text"]),

    _tool("extract_terms",
          "【被动增长引擎】从文本中提取技术术语，自动将不在图谱中的术语注册为 unknown。"
          "每次 AI 回答后调用，让概念图从对话中自动生长。轻量级，无 LLM 参与。",
          {"text": {"type": "string", "description": "要提取术语的文本（通常是 AI 的回答）"}},
          ["text"]),

    _tool("analyze_conversation",
          "【被动推断】分析一轮对话，从用户的消息中推断其对概念的理解水平。"
          "检测：用户展示了理解？用户感到困惑？自动更新概念的 BKT 置信度。"
          "在用户每次消息后调用（不在关键路径上）。",
          {"user_message": {"type": "string", "description": "用户的消息"},
           "ai_response": {"type": "string", "description": "AI 的回答（可选，用于术语提取）",
                           "default": ""}},
          ["user_message"]),

    _tool("get_learning_summary",
          "【Pull 模式入口】获取学习仪表盘摘要：知识统计、最强概念、需关注概念、盲区、遗忘预警。"
          "用户主动调用此工具查看自己的知识状态。",
          {}),

    _tool("mark_known", "标记一个概念为「已理解」。",
          {"term": {"type": "string", "description": "概念标识符"},
           "evidence": {"type": "string", "description": "证据：用户说了什么"}},
          ["term"]),

    _tool("mark_learning", "标记一个概念为「学习中」。",
          {"term": {"type": "string", "description": "概念标识符"},
           "evidence": {"type": "string", "description": "证据"}},
          ["term"]),

    _tool("add_concept", "向知识图谱添加或更新概念。",
          {"id": {"type": "string", "description": "概念 ID（kebab-case）"},
           "name": {"type": "string", "description": "人类可读名称"},
           "explanation": {"type": "string", "description": "1-2 句话的简单解释"},
           "aliases": {"type": "string", "description": "别名，逗号分隔"},
           "parent_ids": {"type": "string", "description": "父概念 ID，逗号分隔"},
           "status": {"type": "string", "description": "状态: known/learning/unknown"},
           "depth_level": {"type": "string", "description": "认知深度: exposed/recall/comprehend/apply/analyze/transfer"}},
          ["id", "name", "explanation"]),

    _tool("get_stats", "获取知识图谱统计（含认知深度分布）。", {}),

    _tool("list_concepts", "列出所有概念，可按状态筛选。",
          {"status": {"type": "string", "description": "筛选: known/learning/unknown"}}),
]

TOOL_MAP = {
    "lookup_concept": tool_lookup_concept,
    "check_answer": tool_check_answer,
    "extract_terms": tool_extract_terms,
    "analyze_conversation": tool_analyze_conversation,
    "get_learning_summary": tool_get_learning_summary,
    "mark_known": tool_mark_known,
    "mark_learning": tool_mark_learning,
    "add_concept": tool_add_concept,
    "get_stats": tool_get_stats,
    "list_concepts": tool_list_concepts,
}


# ── Main ───────────────────────────────────────────────────

def create_server() -> Server:
    server = Server("you-know")

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        fn = TOOL_MAP.get(name)
        if fn is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        result = await fn(**arguments)
        return [TextContent(type="text", text=result)]

    return server


async def main_async():
    store = get_store()
    store.load()
    print(f"📚 you-know v0.2 loaded: {store.graph.stats()}", file=sys.stderr)

    server = create_server()
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main():
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
