"""JSON file-based storage for the knowledge graph."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Concept, KnowledgeGraph, KnowledgeStatus


DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class KnowledgeStore:
    """Persistent storage for a knowledge graph using a local JSON file."""

    def __init__(self, path: Path | str | None = None):
        if path is None:
            path = DEFAULT_DATA_DIR / "knowledge.json"
        self.path = Path(path)
        self.graph: KnowledgeGraph = KnowledgeGraph()

    def load(self) -> KnowledgeGraph:
        """Load the knowledge graph from disk."""
        if not self.path.exists():
            # First run — create seed data with basic concepts
            self.graph = self._create_seed_graph()
            self.save()
            return self.graph

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.graph = KnowledgeGraph.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            self.graph = self._create_seed_graph()
            self.save()

        return self.graph

    def save(self) -> None:
        """Persist the knowledge graph to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.graph.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_concept(self, id: str) -> Concept | None:
        return self.graph.get(id)

    def lookup(self, term: str) -> Concept | None:
        """Find a concept by id, name, or alias."""
        return self.graph.get(term)

    def is_known(self, term: str) -> bool:
        """Check if the user understands a term."""
        return self.graph.is_known(term)

    def mark_known(self, term: str, evidence: str = "") -> Concept | None:
        c = self.graph.mark_known(term, evidence)
        if c:
            self.save()
        return c

    def upsert_concept(
        self,
        id: str,
        name: str,
        explanation: str,
        aliases: list[str] | None = None,
        parent_ids: list[str] | None = None,
        status: str = "unknown",
    ) -> Concept:
        """Create or update a concept."""
        existing = self.graph.get(id)
        if existing:
            existing.name = name
            existing.explanation = explanation
            if aliases is not None:
                existing.aliases = aliases
            if parent_ids is not None:
                existing.parent_ids = parent_ids
            if status != existing.status.value:
                existing.status = KnowledgeStatus(status)
        else:
            existing = Concept(
                id=id,
                name=name,
                explanation=explanation,
                aliases=aliases,
                parent_ids=parent_ids,
                status=KnowledgeStatus(status),
            )
            self.graph.add(existing)
        self.save()
        return existing

    @staticmethod
    def _create_seed_graph() -> KnowledgeGraph:
        """Create the initial knowledge graph with foundational concepts.

        These are concepts the user is CONFIRMED to understand based on
        observed behavior. Everything else is unknown until proven otherwise.
        """
        g = KnowledgeGraph()

        seeds = [
            # --- Confirmed known (from evidence) ---
            Concept(
                id="python",
                name="Python",
                explanation="一门编程语言，用户可以用它写脚本、调用系统命令、做文本处理",
                status=KnowledgeStatus.KNOWN,
                confidence=1.0,
                evidence=["用户写了 old_vs_new.py: subprocess, re, json 等"],
            ),
            Concept(
                id="git",
                name="Git",
                explanation="分布式版本控制系统，用于管理代码变更和协作",
                status=KnowledgeStatus.KNOWN,
                confidence=1.0,
                evidence=["用户项目都是 Git 仓库，日常使用 push/commit/branch"],
            ),
            Concept(
                id="terminal",
                name="终端/命令行",
                explanation="通过文本命令与操作系统交互的界面",
                status=KnowledgeStatus.KNOWN,
                confidence=1.0,
                evidence=["用户熟练使用 bash、环境变量、curl 等"],
            ),
            Concept(
                id="intent-lab",
                name="intent-lab",
                explanation="用户自己构建的跨会话知识管理系统，基于 MCP 协议提供工具接口",
                status=KnowledgeStatus.KNOWN,
                confidence=1.0,
                evidence=["用户创建了 intent-lab 项目"],
            ),
            Concept(
                id="mcp",
                name="MCP (Model Context Protocol)",
                explanation="模型上下文协议 — AI 模型与外部工具/服务之间的标准接口",
                status=KnowledgeStatus.KNOWN,
                confidence=0.9,
                evidence=["用户的 intent-lab 项目通过 MCP 暴露工具"],
            ),
            Concept(
                id="agentic-design-patterns",
                name="Agentic Design Patterns",
                explanation="AI Agent 设计模式 — 包括 Prompt Chaining、Routing、Reflection、Parallelization",
                status=KnowledgeStatus.LEARNING,
                confidence=0.7,
                evidence=["用户正在系统学习 Grok Build 源码中的这些模式"],
            ),
            Concept(
                id="grok-build",
                name="Grok Build",
                explanation="SpaceXAI 的终端 AI 编程助手，Rust 编写的大型项目",
                status=KnowledgeStatus.LEARNING,
                confidence=0.8,
                evidence=["用户正在学习其源码"],
            ),

            # --- Known to be unknown (try to understand) ---
            Concept(
                id="rust",
                name="Rust",
                explanation="一门系统编程语言，注重内存安全和性能。Grok Build 就是 Rust 写的",
                aliases=["rustc", "cargo"],
                status=KnowledgeStatus.UNKNOWN,
            ),
            Concept(
                id="claude-code-hook",
                name="Claude Code Hook",
                explanation="Claude Code 的事件钩子，可在会话启动/结束时自动执行脚本，配置在 settings.json 里",
                aliases=["hook", "hooks"],
                parent_ids=["claude-code"],
                status=KnowledgeStatus.LEARNING,
                evidence=["刚刚配置完 grok-build 的 hooks"],
            ),
            Concept(
                id="claude-code",
                name="Claude Code",
                explanation="Anthropic 的终端 AI 编程助手 CLI 工具",
                aliases=["claude", "claude-cli"],
                status=KnowledgeStatus.KNOWN,
                confidence=1.0,
                evidence=["用户每天都在使用"],
            ),
            Concept(
                id="claude-code-settings",
                name="Claude Code settings.json",
                explanation="Claude Code 的项目配置文件，定义 hooks、权限、MCP 服务器等",
                aliases=["settings.json", "claude settings"],
                parent_ids=["claude-code"],
                status=KnowledgeStatus.LEARNING,
            ),
            Concept(
                id="claude-code-memory",
                name="Claude Code Memory 系统",
                explanation="Claude Code 的持久化记忆，按项目存储，AI 会话间共享。自动加载 MEMORY.md 索引下的文件",
                aliases=["memory", "session-context"],
                parent_ids=["claude-code"],
                status=KnowledgeStatus.LEARNING,
            ),
            Concept(
                id="session-context",
                name="session-context.md",
                explanation="由 sync-intent-lab.py 自动生成的上下文快照，包含上次会话的状态、决策、下一步建议",
                aliases=["会话上下文"],
                parent_ids=["intent-lab", "claude-code-memory"],
                status=KnowledgeStatus.LEARNING,
            ),
            Concept(
                id="bootstrap",
                name="Bootstrap (启动加载)",
                explanation="系统/会话启动时自动加载初始状态的过程。如电脑开机加载操作系统、AI 会话启动加载上次上下文",
                aliases=["启动初始化", "自动加载"],
                status=KnowledgeStatus.LEARNING,
            ),
            Concept(
                id="reflection-pattern",
                name="Reflection (反思) 模式",
                explanation="Agentic Design Pattern 之一 — Agent 自我检查产出、发现差距、修正。包含 Verifier → gap feedback → strategist 循环",
                aliases=["反思模式", "Reflection"],
                parent_ids=["agentic-design-patterns"],
                status=KnowledgeStatus.LEARNING,
            ),
            Concept(
                id="recursive-explanation",
                name="递归解释循环",
                explanation="回答用户问题时，检查回答中的术语用户是否理解。不理解就展开解释，再检查展开后的解释是否又有不理解的词，递归直到全部清楚",
                aliases=["递归解释"],
                parent_ids=["you-know"],
                status=KnowledgeStatus.LEARNING,
            ),
            Concept(
                id="you-know",
                name="you-know (知道么)",
                explanation="用户正在构建的个人知识模型系统 — AI 回答前先查「你懂不懂这些词」，不懂就先解释。AI 无关，通过 MCP 协议供任何 AI 使用",
                aliases=["知道么", "youknow"],
                status=KnowledgeStatus.LEARNING,
            ),
            Concept(
                id="knowledge-graph",
                name="知识图谱",
                explanation="一种用节点和边表示知识的数据结构。这里是用户理解的技术概念之间的关系网络",
                aliases=["knowledge graph", "概念图"],
                parent_ids=["you-know"],
                status=KnowledgeStatus.LEARNING,
            ),
            Concept(
                id="sqlite",
                name="SQLite",
                explanation="轻量级嵌入式数据库，存在单个文件里。intent-lab 用它存储工作事件和决策",
                status=KnowledgeStatus.UNKNOWN,
            ),
            Concept(
                id="student-model",
                name="学生模型 (Student Model)",
                explanation="智能教学系统（ITS）的核心组件 — 代表系统对学生的知识状态的理解。你的 you-know 就是一个 AI 无关的 Student Model",
                aliases=["Student Model", "学习者模型"],
                parent_ids=["you-know"],
                status=KnowledgeStatus.UNKNOWN,
            ),
        ]

        for s in seeds:
            g.add(s)

        return g
