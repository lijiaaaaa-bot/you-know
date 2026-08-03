"""Data models for the personal knowledge graph."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class KnowledgeStatus(str, Enum):
    KNOWN = "known"        # User confirmed they understand this
    LEARNING = "learning"  # Currently learning, partial understanding
    UNKNOWN = "unknown"    # Not yet understood


class Concept:
    """A single concept in the user's knowledge model."""

    def __init__(
        self,
        id: str,                        # kebab-case, e.g. "claude-code-hook"
        name: str,                      # Human-readable, e.g. "Claude Code Hook"
        explanation: str,               # 1-2 sentence simple explanation
        aliases: list[str] | None = None,
        parent_ids: list[str] | None = None,  # Parent concepts (broader topics)
        status: KnowledgeStatus = KnowledgeStatus.UNKNOWN,
        confidence: float = 0.0,        # 0.0-1.0, how sure we are about this status
        evidence: list[str] | None = None,  # Why we think this status is correct
        last_updated: str = "",
    ):
        self.id = id
        self.name = name
        self.explanation = explanation
        self.aliases = aliases or []
        self.parent_ids = parent_ids or []
        self.status = status
        self.confidence = confidence
        self.evidence = evidence or []
        self.last_updated = last_updated or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "explanation": self.explanation,
            "aliases": self.aliases,
            "parent_ids": self.parent_ids,
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Concept":
        return cls(
            id=d["id"],
            name=d["name"],
            explanation=d["explanation"],
            aliases=d.get("aliases", []),
            parent_ids=d.get("parent_ids", []),
            status=KnowledgeStatus(d.get("status", "unknown")),
            confidence=d.get("confidence", 0.0),
            evidence=d.get("evidence", []),
            last_updated=d.get("last_updated", ""),
        )

    def is_known(self) -> bool:
        return self.status == KnowledgeStatus.KNOWN

    def is_unknown(self) -> bool:
        return self.status == KnowledgeStatus.UNKNOWN

    def __repr__(self) -> str:
        return f"Concept(id={self.id!r}, status={self.status.value})"


class KnowledgeGraph:
    """A graph of concepts representing what a user understands."""

    def __init__(self, concepts: dict[str, Concept] | None = None):
        self.concepts: dict[str, Concept] = concepts or {}

    def get(self, id: str) -> Optional[Concept]:
        """Get a concept by id or alias."""
        if id in self.concepts:
            return self.concepts[id]
        for c in self.concepts.values():
            if id in c.aliases:
                return c
        return None

    def is_known(self, id: str) -> bool:
        """Check if a concept is known (also checks aliases and parents)."""
        c = self.get(id)
        if c is None:
            return False
        if c.is_known():
            return True
        # If any parent is known, this is likely known
        for pid in c.parent_ids:
            if self.is_known(pid):
                return True
        return False

    def is_unknown(self, id: str) -> bool:
        """Check if a concept is explicitly unknown."""
        c = self.get(id)
        if c is None:
            return True  # Not in graph = unknown
        if c.is_unknown():
            return True
        # If all parents are unknown, this is unknown
        if c.parent_ids and all(self.is_unknown(pid) for pid in c.parent_ids):
            return True
        return False

    def add(self, concept: Concept) -> None:
        self.concepts[concept.id] = concept

    def mark_known(self, id: str, evidence: str = "") -> Optional[Concept]:
        """Mark a concept as known."""
        c = self.get(id)
        if c is None:
            return None
        c.status = KnowledgeStatus.KNOWN
        c.confidence = 1.0
        if evidence:
            c.evidence.append(evidence)
        c.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return c

    def mark_learning(self, id: str, evidence: str = "") -> Optional[Concept]:
        c = self.get(id)
        if c is None:
            return None
        c.status = KnowledgeStatus.LEARNING
        c.confidence = 0.5
        if evidence:
            c.evidence.append(evidence)
        c.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return c

    def get_unknown_for_text(self, text: str) -> list[Concept]:
        """Find concepts mentioned in text that the user doesn't understand.

        This is a simple substring match — the caller (AI) should extract
        key terms and pass them here.
        """
        unknown = []
        for c in self.concepts.values():
            if c.is_unknown():
                # Check if this concept or its aliases appear in text
                terms = [c.name] + c.aliases
                if any(t.lower() in text.lower() for t in terms):
                    unknown.append(c)
        return unknown

    def all_known(self) -> list[Concept]:
        return [c for c in self.concepts.values() if c.is_known()]

    def all_unknown(self) -> list[Concept]:
        return [c for c in self.concepts.values() if c.is_unknown()]

    def stats(self) -> dict:
        known = sum(1 for c in self.concepts.values() if c.is_known())
        learning = sum(1 for c in self.concepts.values() if c.status == KnowledgeStatus.LEARNING)
        unknown = sum(1 for c in self.concepts.values() if c.is_unknown())
        return {
            "total": len(self.concepts),
            "known": known,
            "learning": learning,
            "unknown": unknown,
        }

    def to_dict(self) -> dict:
        return {
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "stats": self.stats(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeGraph":
        concepts = {}
        for k, v in d.get("concepts", {}).items():
            concepts[k] = Concept.from_dict(v)
        return cls(concepts)
