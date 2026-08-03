"""Storage backends for the knowledge graph.

v0.2 — Dual backend:
  - KnowledgeStore: JSON file (default, < 500 concepts)
  - SQLiteStore: SQLite + FTS5 (auto-selected when concepts > 500 or explicitly chosen)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import (
    Concept,
    DepthLevel,
    EvidenceQuality,
    KnowledgeGraph,
    KnowledgeStatus,
)

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ── JSON file backend ──────────────────────────────────────

class KnowledgeStore:
    """Persistent storage for a knowledge graph using a local JSON file.

    Good for < 500 concepts. Switch to SQLiteStore for larger graphs.
    """

    def __init__(self, path: Path | str | None = None):
        if path is None:
            path = DEFAULT_DATA_DIR / "knowledge.json"
        self.path = Path(path)
        self.graph: KnowledgeGraph = KnowledgeGraph()

    def load(self) -> KnowledgeGraph:
        """Load the knowledge graph from disk."""
        if not self.path.exists():
            self.graph = self._create_seed_graph()
            self.save()
            return self.graph

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.graph = KnowledgeGraph.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            self.graph = self._create_seed_graph()
            self.save()

        # Apply forgetting curve decay on load
        self.graph.apply_decay_all()
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
                id=id, name=name, explanation=explanation,
                aliases=aliases, parent_ids=parent_ids,
                status=KnowledgeStatus(status),
            )
            self.graph.add(existing)
        self.save()
        return existing

    # ── Seed data ──────────────────────────────────────

    @staticmethod
    def _create_seed_graph() -> KnowledgeGraph:
        """Create the initial knowledge graph with foundational concepts."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        g = KnowledgeGraph()

        seeds = [
            Concept(
                id="python", name="Python",
                explanation="一门编程语言，用户可以用它写脚本、调用系统命令、做文本处理",
                status=KnowledgeStatus.KNOWN, confidence=1.0,
                depth_level=DepthLevel.APPLY,
                evidence_quality=EvidenceQuality.BEHAVIORAL,
                stability=0.9, last_retrieval=now,
                evidence=["用户写了 old_vs_new.py: subprocess, re, json 等"],
            ),
            Concept(
                id="git", name="Git",
                explanation="分布式版本控制系统，用于管理代码变更和协作",
                status=KnowledgeStatus.KNOWN, confidence=1.0,
                depth_level=DepthLevel.APPLY,
                evidence_quality=EvidenceQuality.BEHAVIORAL,
                stability=0.9, last_retrieval=now,
                evidence=["用户项目都是 Git 仓库，日常使用 push/commit/branch"],
            ),
            Concept(
                id="terminal", name="终端/命令行",
                explanation="通过文本命令与操作系统交互的界面",
                status=KnowledgeStatus.KNOWN, confidence=1.0,
                depth_level=DepthLevel.APPLY,
                evidence_quality=EvidenceQuality.BEHAVIORAL,
                stability=0.9, last_retrieval=now,
                evidence=["用户熟练使用 bash、环境变量、curl 等"],
            ),
            Concept(
                id="intent-lab", name="intent-lab",
                explanation="用户自己构建的跨会话知识管理系统，基于 MCP 协议提供工具接口",
                status=KnowledgeStatus.KNOWN, confidence=1.0,
                depth_level=DepthLevel.TRANSFER,
                evidence_quality=EvidenceQuality.BEHAVIORAL,
                stability=0.95, last_retrieval=now,
                evidence=["用户创建了 intent-lab 项目"],
            ),
            Concept(
                id="mcp", name="MCP (Model Context Protocol)",
                explanation="模型上下文协议 — AI 模型与外部工具/服务之间的标准接口",
                status=KnowledgeStatus.KNOWN, confidence=0.95,
                depth_level=DepthLevel.COMPREHEND,
                evidence_quality=EvidenceQuality.BEHAVIORAL,
                stability=0.85, last_retrieval=now,
                evidence=["用户的 intent-lab 项目通过 MCP 暴露工具"],
            ),
            Concept(
                id="agentic-design-patterns", name="Agentic Design Patterns",
                explanation="AI Agent 设计模式 — 包括 Prompt Chaining、Routing、Reflection、Parallelization",
                status=KnowledgeStatus.LEARNING, confidence=0.7,
                depth_level=DepthLevel.COMPREHEND,
                evidence_quality=EvidenceQuality.BEHAVIORAL,
                stability=0.4, last_retrieval=now,
                evidence=["用户正在系统学习 Grok Build 源码中的这些模式"],
            ),
            Concept(
                id="grok-build", name="Grok Build",
                explanation="SpaceXAI 的终端 AI 编程助手，Rust 编写的大型项目",
                status=KnowledgeStatus.LEARNING, confidence=0.8,
                depth_level=DepthLevel.RECALL,
                evidence_quality=EvidenceQuality.BEHAVIORAL,
                stability=0.3, last_retrieval=now,
                evidence=["用户正在学习其源码"],
            ),
            Concept(
                id="rust", name="Rust",
                explanation="一门系统编程语言，注重内存安全和性能。Grok Build 就是 Rust 写的",
                aliases=["rustc", "cargo"],
                status=KnowledgeStatus.UNKNOWN, confidence=0.15,
                depth_level=DepthLevel.EXPOSED,
            ),
            Concept(
                id="claude-code-hook", name="Claude Code Hook",
                explanation="Claude Code 的事件钩子，可在会话启动/结束时自动执行脚本",
                aliases=["hook", "hooks"],
                parent_ids=["claude-code"],
                status=KnowledgeStatus.LEARNING, confidence=0.55,
                depth_level=DepthLevel.RECALL,
                stability=0.35, last_retrieval=now,
            ),
            Concept(
                id="claude-code", name="Claude Code",
                explanation="Anthropic 的终端 AI 编程助手 CLI 工具",
                aliases=["claude", "claude-cli"],
                status=KnowledgeStatus.KNOWN, confidence=1.0,
                depth_level=DepthLevel.APPLY,
                evidence_quality=EvidenceQuality.BEHAVIORAL,
                stability=0.9, last_retrieval=now,
                evidence=["用户每天都在使用"],
            ),
            Concept(
                id="claude-code-settings", name="Claude Code settings.json",
                explanation="Claude Code 的项目配置文件，定义 hooks、权限、MCP 服务器等",
                aliases=["settings.json", "claude settings"],
                parent_ids=["claude-code"],
                status=KnowledgeStatus.LEARNING, confidence=0.55,
                depth_level=DepthLevel.RECALL,
                stability=0.3, last_retrieval=now,
            ),
            Concept(
                id="claude-code-memory", name="Claude Code Memory 系统",
                explanation="Claude Code 的持久化记忆，按项目存储，AI 会话间共享",
                aliases=["memory", "session-context"],
                parent_ids=["claude-code"],
                status=KnowledgeStatus.LEARNING, confidence=0.55,
                depth_level=DepthLevel.RECALL,
                stability=0.3, last_retrieval=now,
            ),
            Concept(
                id="session-context", name="session-context.md",
                explanation="由 sync-intent-lab.py 自动生成的上下文快照",
                aliases=["会话上下文"],
                parent_ids=["intent-lab", "claude-code-memory"],
                status=KnowledgeStatus.LEARNING, confidence=0.5,
                depth_level=DepthLevel.RECALL,
                stability=0.25, last_retrieval=now,
            ),
            Concept(
                id="bootstrap", name="Bootstrap (启动加载)",
                explanation="系统/会话启动时自动加载初始状态的过程",
                aliases=["启动初始化", "自动加载"],
                status=KnowledgeStatus.LEARNING, confidence=0.5,
                depth_level=DepthLevel.COMPREHEND,
                stability=0.3, last_retrieval=now,
            ),
            Concept(
                id="reflection-pattern", name="Reflection (反思) 模式",
                explanation="Agentic Design Pattern — Agent 自我检查产出、发现差距、修正",
                aliases=["反思模式", "Reflection"],
                parent_ids=["agentic-design-patterns"],
                status=KnowledgeStatus.LEARNING, confidence=0.55,
                depth_level=DepthLevel.COMPREHEND,
                stability=0.25, last_retrieval=now,
            ),
            Concept(
                id="recursive-explanation", name="递归解释循环",
                explanation="回答中检查术语用户是否理解，不理解就展开，递归直到清楚",
                aliases=["递归解释"],
                parent_ids=["you-know"],
                status=KnowledgeStatus.LEARNING, confidence=0.6,
                depth_level=DepthLevel.COMPREHEND,
                stability=0.3, last_retrieval=now,
            ),
            Concept(
                id="you-know", name="you-know (知道么)",
                explanation="个人知识模型系统 — AI 回答前先查用户是否理解相关概念",
                aliases=["知道么", "youknow"],
                status=KnowledgeStatus.LEARNING, confidence=0.65,
                depth_level=DepthLevel.TRANSFER,
                stability=0.4, last_retrieval=now,
            ),
            Concept(
                id="knowledge-graph", name="知识图谱",
                explanation="用节点和边表示知识的数据结构 — 这里是技术概念关系网络",
                aliases=["knowledge graph", "概念图"],
                parent_ids=["you-know"],
                status=KnowledgeStatus.LEARNING, confidence=0.55,
                depth_level=DepthLevel.COMPREHEND,
                stability=0.3, last_retrieval=now,
            ),
            Concept(
                id="student-model", name="学生模型 (Student Model)",
                explanation="智能教学系统核心组件 — 代表系统对学生知识状态的理解",
                aliases=["Student Model", "学习者模型"],
                parent_ids=["you-know"],
                status=KnowledgeStatus.UNKNOWN, confidence=0.1,
                depth_level=DepthLevel.EXPOSED,
            ),
        ]

        for s in seeds:
            g.add(s)

        return g


# ── SQLite + FTS5 backend ──────────────────────────────────

class SQLiteStore:
    """SQLite-based storage with FTS5 full-text search.

    Use when concept count > 500 or when concurrent read/write is needed.
    Same interface as KnowledgeStore.
    """

    def __init__(self, path: Path | str | None = None):
        if path is None:
            path = DEFAULT_DATA_DIR / "knowledge.db"
        self.path = Path(path)
        self.graph: KnowledgeGraph = KnowledgeGraph()
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Create tables if not exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                explanation TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL NOT NULL DEFAULT 0.0,
                depth_level TEXT NOT NULL DEFAULT 'exposed',
                evidence_quality TEXT NOT NULL DEFAULT 'auto',
                last_retrieval TEXT NOT NULL DEFAULT '',
                stability REAL NOT NULL DEFAULT 0.0,
                last_updated TEXT NOT NULL DEFAULT '',
                extra_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts
                USING fts5(id, name, explanation, content='concepts',
                           content_rowid='rowid');

            CREATE TABLE IF NOT EXISTS aliases (
                alias TEXT PRIMARY KEY,
                concept_id TEXT NOT NULL REFERENCES concepts(id)
            );

            CREATE TABLE IF NOT EXISTS parents (
                concept_id TEXT NOT NULL REFERENCES concepts(id),
                parent_id TEXT NOT NULL REFERENCES concepts(id),
                PRIMARY KEY (concept_id, parent_id)
            );

            CREATE INDEX IF NOT EXISTS idx_concepts_status ON concepts(status);
            CREATE INDEX IF NOT EXISTS idx_concepts_confidence ON concepts(confidence);
            CREATE INDEX IF NOT EXISTS idx_aliases_concept ON aliases(concept_id);
        """)

    def load(self) -> KnowledgeGraph:
        """Load all concepts from SQLite into a KnowledgeGraph in memory."""
        try:
            rows = self.conn.execute(
                "SELECT * FROM concepts ORDER BY confidence DESC"
            ).fetchall()

            g = KnowledgeGraph()
            for row in rows:
                c = Concept(
                    id=row["id"],
                    name=row["name"],
                    explanation=row["explanation"],
                    aliases=self._load_aliases(row["id"]),
                    parent_ids=self._load_parents(row["id"]),
                    status=KnowledgeStatus(row["status"]),
                    confidence=row["confidence"],
                    depth_level=DepthLevel(row["depth_level"]),
                    evidence_quality=EvidenceQuality(row["evidence_quality"]),
                    last_retrieval=row["last_retrieval"],
                    stability=row["stability"],
                    last_updated=row["last_updated"],
                )
                g.add(c)

            self.graph = g
            g.apply_decay_all()
            return g

        except sqlite3.OperationalError:
            # Fresh database
            json_store = KnowledgeStore()
            json_store.load()
            self.graph = json_store.graph
            self._save_all()
            return self.graph

    def _load_aliases(self, concept_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT alias FROM aliases WHERE concept_id = ?", (concept_id,)
        ).fetchall()
        return [r["alias"] for r in rows]

    def _load_parents(self, concept_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT parent_id FROM parents WHERE concept_id = ?", (concept_id,)
        ).fetchall()
        return [r["parent_id"] for r in rows]

    def save(self) -> None:
        """Persist all concepts to SQLite."""
        self._save_all()

    def _save_all(self) -> None:
        """Full save — used for bulk operations."""
        with self.conn:
            self.conn.execute("DELETE FROM concepts")
            self.conn.execute("DELETE FROM aliases")
            self.conn.execute("DELETE FROM parents")
            self.conn.execute("DELETE FROM concepts_fts")

            for c in self.graph.concepts.values():
                self.conn.execute(
                    """INSERT INTO concepts
                       (id, name, explanation, status, confidence,
                        depth_level, evidence_quality,
                        last_retrieval, stability, last_updated, extra_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (c.id, c.name, c.explanation, c.status.value,
                     c.confidence, c.depth_level.value,
                     c.evidence_quality.value,
                     c.last_retrieval, c.stability, c.last_updated,
                     json.dumps({"evidence": c.evidence})),
                )
                for alias in c.aliases:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO aliases (alias, concept_id) VALUES (?, ?)",
                        (alias, c.id),
                    )
                for pid in c.parent_ids:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO parents (concept_id, parent_id) VALUES (?, ?)",
                        (c.id, pid),
                    )
                # FTS index
                self.conn.execute(
                    "INSERT INTO concepts_fts (id, name, explanation) VALUES (?, ?, ?)",
                    (c.id, c.name, c.explanation),
                )

    def get_concept(self, id: str) -> Concept | None:
        return self.graph.get(id)

    def lookup(self, term: str) -> Concept | None:
        return self.graph.get(term)

    def is_known(self, term: str) -> bool:
        return self.graph.is_known(term)

    def mark_known(self, term: str, evidence: str = "") -> Concept | None:
        c = self.graph.mark_known(term, evidence)
        if c:
            self._save_concept(c)
        return c

    def _save_concept(self, c: Concept) -> None:
        """Save a single concept (row-level update)."""
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO concepts
                   (id, name, explanation, status, confidence,
                    depth_level, evidence_quality,
                    last_retrieval, stability, last_updated, extra_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (c.id, c.name, c.explanation, c.status.value,
                 c.confidence, c.depth_level.value,
                 c.evidence_quality.value,
                 c.last_retrieval, c.stability, c.last_updated,
                 json.dumps({"evidence": c.evidence})),
            )
            # Update FTS
            self.conn.execute(
                "INSERT OR REPLACE INTO concepts_fts (rowid, id, name, explanation) "
                "VALUES ((SELECT rowid FROM concepts WHERE id=?), ?, ?, ?)",
                (c.id, c.id, c.name, c.explanation),
            )

    def upsert_concept(
        self, id: str, name: str, explanation: str,
        aliases: list[str] | None = None,
        parent_ids: list[str] | None = None,
        status: str = "unknown",
    ) -> Concept:
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
                id=id, name=name, explanation=explanation,
                aliases=aliases, parent_ids=parent_ids,
                status=KnowledgeStatus(status),
            )
            self.graph.add(existing)
        self._save_concept(existing)
        return existing

    def search_fts(self, query: str, limit: int = 10) -> list[Concept]:
        """Full-text search across concept names and explanations."""
        rows = self.conn.execute(
            """SELECT c.* FROM concepts c
               JOIN concepts_fts f ON c.id = f.id
               WHERE concepts_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()

        results = []
        for row in rows:
            c = self.graph.get(row["id"])
            if c:
                results.append(c)
        return results
