"""MCP server for you-know — expose the knowledge model to any AI via MCP.

Tools:
  - lookup_concept: Check if user knows a concept, get explanation if not
  - check_answer: Analyze an answer text for unknown concepts, recursively expand
  - mark_known: Mark a concept as known by the user
  - add_concept: Add a new concept to the knowledge graph
  - get_stats: Get knowledge graph statistics

Run: python -m you_know.server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .engine import check_answer
from .store import KnowledgeStore

# ── Globals ────────────────────────────────────────────────
_store: KnowledgeStore | None = None
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store


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
            "suggestion": "Concept not in graph. Use add_concept to register it, or check the spelling.",
        }, ensure_ascii=False)

    return json.dumps({
        "found": True,
        "id": concept.id,
        "name": concept.name,
        "explanation": concept.explanation,
        "status": concept.status.value,
        "confidence": concept.confidence,
        "aliases": concept.aliases,
        "parent_ids": concept.parent_ids,
        "is_known": concept.is_known(),
    }, ensure_ascii=False)


async def tool_check_answer(text: str, max_depth: int = 3) -> str:
    """Check an answer text. Returns unknown concepts and optionally an expanded version."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    result = check_answer(text, store, max_depth=max_depth)
    return json.dumps(result, ensure_ascii=False, indent=2)


async def tool_mark_known(term: str, evidence: str = "") -> str:
    """Mark a concept as understood by the user."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    concept = store.mark_known(term, evidence)
    if concept is None:
        # Maybe it doesn't exist yet — auto-create it
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

    return json.dumps({
        "ok": True,
        "id": concept.id,
        "name": concept.name,
        "status": concept.status.value,
    }, ensure_ascii=False)


async def tool_add_concept(
    id: str,
    name: str,
    explanation: str,
    aliases: str = "",
    parent_ids: str = "",
    status: str = "unknown",
) -> str:
    """Add or update a concept in the knowledge graph."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    alias_list = [a.strip() for a in aliases.split(",") if a.strip()] if aliases else []
    parent_list = [p.strip() for p in parent_ids.split(",") if p.strip()] if parent_ids else []

    concept = store.upsert_concept(
        id=id,
        name=name,
        explanation=explanation,
        aliases=alias_list,
        parent_ids=parent_list,
        status=status,
    )

    return json.dumps({
        "ok": True,
        "id": concept.id,
        "name": concept.name,
        "status": concept.status.value,
    }, ensure_ascii=False)


async def tool_get_stats() -> str:
    """Get knowledge graph statistics."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    stats = store.graph.stats()
    return json.dumps(stats, ensure_ascii=False)


async def tool_list_concepts(status: str = "") -> str:
    """List concepts, optionally filtered by status (known/learning/unknown)."""
    store = get_store()
    if not store.graph.concepts:
        store.load()

    if status:
        concepts = [c for c in store.graph.concepts.values() if c.status.value == status]
    else:
        concepts = list(store.graph.concepts.values())

    return json.dumps(
        [{"id": c.id, "name": c.name, "status": c.status.value} for c in concepts],
        ensure_ascii=False,
        indent=2,
    )


# ── Server setup ───────────────────────────────────────────

TOOLS = [
    Tool(
        name="lookup_concept",
        description="查询一个概念：用户是否理解它？如果不理解，返回简单解释。用 id、名称或别名查找。",
        inputSchema={
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "概念标识符（id、名称或别名），如 'claude-code-hook' 或 'hook'",
                },
            },
            "required": ["term"],
        },
    ),
    Tool(
        name="check_answer",
        description="分析一段回答文本，找出用户不理解的术语，递归展开解释。这是核心工具——AI 回答用户问题前应该先调这个。",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要检查的回答文本",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "递归解释的最大深度（默认 3）",
                    "default": 3,
                },
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="mark_known",
        description="标记一个概念为「用户已理解」。当用户明确表示他们理解某个概念时调用。",
        inputSchema={
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "概念标识符",
                },
                "evidence": {
                    "type": "string",
                    "description": "证据：用户说了什么让你判断他理解这个概念",
                },
            },
            "required": ["term"],
        },
    ),
    Tool(
        name="mark_learning",
        description="标记一个概念为「学习中」——用户正在理解但尚未完全掌握。",
        inputSchema={
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "概念标识符",
                },
                "evidence": {
                    "type": "string",
                    "description": "证据",
                },
            },
            "required": ["term"],
        },
    ),
    Tool(
        name="add_concept",
        description="向知识图谱添加新概念或更新已有概念。",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "概念 ID（kebab-case），如 'python-asyncio'",
                },
                "name": {
                    "type": "string",
                    "description": "人类可读的名称，如 'Python asyncio'",
                },
                "explanation": {
                    "type": "string",
                    "description": "1-2 句话的简单解释",
                },
                "aliases": {
                    "type": "string",
                    "description": "别名，逗号分隔，如 'asyncio,async/await'",
                },
                "parent_ids": {
                    "type": "string",
                    "description": "父概念 ID，逗号分隔，如 'python,concurrency'",
                },
                "status": {
                    "type": "string",
                    "description": "状态: known, learning, unknown（默认 unknown）",
                },
            },
            "required": ["id", "name", "explanation"],
        },
    ),
    Tool(
        name="get_stats",
        description="获取知识图谱统计：总共多少概念、多少已知、多少学习中、多少未知。",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="list_concepts",
        description="列出所有概念，可按状态筛选。",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "筛选状态: known, learning, unknown（空 = 全部）",
                },
            },
        },
    ),
]

TOOL_MAP = {
    "lookup_concept": tool_lookup_concept,
    "check_answer": tool_check_answer,
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
    print(f"📚 you-know loaded: {store.graph.stats()}", file=sys.stderr)

    server = create_server()
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main():
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
