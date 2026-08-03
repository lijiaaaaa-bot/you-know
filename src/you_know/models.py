"""Data models for the personal knowledge graph.

v0.2 — Enhanced with:
  - Bloom depth levels (exposed → transfer)
  - BKT-style confidence with Bayesian updating
  - Ebbinghaus forgetting curve support (stability, last_retrieval)
  - Evidence quality tracking
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class KnowledgeStatus(str, Enum):
    KNOWN = "known"        # User confirmed they understand this
    LEARNING = "learning"  # Currently learning, partial understanding
    UNKNOWN = "unknown"    # Not yet understood


class DepthLevel(str, Enum):
    """Bloom's revised taxonomy — cognitive depth of understanding."""
    EXPOSED = "exposed"        # Saw explanation, can't recall on own
    RECALL = "recall"           # Can recall with hints
    COMPREHEND = "comprehend"   # Can explain in own words
    APPLY = "apply"             # Can use correctly in context
    ANALYZE = "analyze"         # Can break down and evaluate
    TRANSFER = "transfer"       # Can teach others / apply in new domains


class EvidenceQuality(str, Enum):
    """How reliable is the evidence for this concept's status?"""
    EXPLICIT = "explicit"       # User explicitly confirmed (highest)
    BEHAVIORAL = "behavioral"   # Inferred from user behavior (medium)
    HEURISTIC = "heuristic"     # Rule-based inference (low)
    AUTO = "auto"               # Automatic extraction, unverified (lowest)


class Concept:
    """A single concept in the user's knowledge model.

    v0.2 adds depth_level, forgetting curve support, and evidence tracking.
    """

    def __init__(
        self,
        id: str,                        # kebab-case, e.g. "claude-code-hook"
        name: str,                      # Human-readable, e.g. "Claude Code Hook"
        explanation: str,               # 1-2 sentence simple explanation
        aliases: list[str] | None = None,
        parent_ids: list[str] | None = None,  # Parent concepts (broader topics)
        status: KnowledgeStatus = KnowledgeStatus.UNKNOWN,
        confidence: float = 0.0,        # P(known) — BKT-style continuous probability
        depth_level: DepthLevel = DepthLevel.EXPOSED,
        evidence: list[str] | None = None,
        evidence_quality: EvidenceQuality = EvidenceQuality.AUTO,
        last_retrieval: str = "",        # ISO timestamp of last access
        stability: float = 0.0,          # Memory stability (Anki-like SRS)
        last_updated: str = "",
    ):
        self.id = id
        self.name = name
        self.explanation = explanation
        self.aliases = aliases or []
        self.parent_ids = parent_ids or []
        self.status = status
        self.confidence = confidence
        self.depth_level = depth_level
        self.evidence = evidence or []
        self.evidence_quality = evidence_quality
        self.last_retrieval = last_retrieval
        self.stability = stability
        self.last_updated = last_updated or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    # ── BKT-style confidence update ──────────────────────

    def update_confidence(
        self,
        evidence_signal: float,  # +1.0 = strong positive, -1.0 = strong negative
        quality: EvidenceQuality = EvidenceQuality.BEHAVIORAL,
    ) -> float:
        """Bayesian-style confidence update.

        Uses a simplified BKT (Bayesian Knowledge Tracing) model:
          P_new = P_old + learning_rate * (evidence - P_old)

        where evidence is scaled to [0, 1] from the signal.
        The learning_rate varies by evidence quality.
        """
        quality_weights = {
            EvidenceQuality.EXPLICIT: 0.3,    # User said "I know this" — strong
            EvidenceQuality.BEHAVIORAL: 0.15,  # User demonstrated — medium
            EvidenceQuality.HEURISTIC: 0.08,   # Rule-based guess — weak
            EvidenceQuality.AUTO: 0.03,         # Auto-extracted — very weak
        }
        learning_rate = quality_weights.get(quality, 0.05)

        # Map signal [-1, 1] to evidence [0, 1]
        evidence_val = (evidence_signal + 1.0) / 2.0  # now 0..1
        self.confidence += learning_rate * (evidence_val - self.confidence)
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Auto-transition status based on confidence thresholds
        if self.confidence >= 0.85:
            self.status = KnowledgeStatus.KNOWN
            if self.depth_level in (DepthLevel.EXPOSED,):
                self.depth_level = DepthLevel.RECALL
        elif self.confidence >= 0.35:
            self.status = KnowledgeStatus.LEARNING
        else:
            self.status = KnowledgeStatus.UNKNOWN

        self.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.confidence

    # ── Forgetting curve ──────────────────────────────────

    def apply_decay(self, reference_time: datetime | None = None) -> float:
        """Apply Ebbinghaus forgetting curve decay to confidence.

        Uses a simplified exponential decay model:
          S(t) = stability * e^(-t / halflife)

        where halflife depends on depth_level (deeper = slower decay).
        Returns the decayed stability, and updates confidence accordingly.
        """
        if not self.last_retrieval or self.stability <= 0:
            return self.confidence

        now = reference_time or datetime.now(timezone.utc)
        last = datetime.fromisoformat(
            self.last_retrieval.replace("Z", "+00:00")
        )
        elapsed_days = (now - last).total_seconds() / 86400.0

        if elapsed_days <= 0:
            return self.confidence

        # Halflife depends on depth: deeper understanding decays slower
        depth_halflife = {
            DepthLevel.EXPOSED: 3,       # ~3 days
            DepthLevel.RECALL: 7,         # ~1 week
            DepthLevel.COMPREHEND: 21,    # ~3 weeks
            DepthLevel.APPLY: 60,         # ~2 months
            DepthLevel.ANALYZE: 120,      # ~4 months
            DepthLevel.TRANSFER: 365,     # ~1 year
        }
        halflife = depth_halflife.get(self.depth_level, 7)

        decay_factor = math.exp(-elapsed_days / halflife)
        self.stability *= decay_factor
        self.stability = max(0.0, min(1.0, self.stability))

        # Blend stability into confidence for a smoothed estimate
        self.confidence = 0.3 * self.confidence + 0.7 * self.stability
        self.confidence = max(0.0, min(1.0, self.confidence))

        return self.confidence

    def record_retrieval(self) -> None:
        """Record a retrieval event — resets the forgetting curve clock."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.last_retrieval = now
        self.last_updated = now
        # Boost stability on retrieval (spacing effect)
        self.stability = min(1.0, self.stability + 0.05)

    # ── Status helpers ────────────────────────────────────

    def is_known(self) -> bool:
        return self.status == KnowledgeStatus.KNOWN

    def is_unknown(self) -> bool:
        return self.status == KnowledgeStatus.UNKNOWN

    def needs_explanation(self) -> bool:
        """Should this concept be explained to the user?"""
        return self.status in (KnowledgeStatus.UNKNOWN, KnowledgeStatus.LEARNING)

    def __repr__(self) -> str:
        return (
            f"Concept(id={self.id!r}, status={self.status.value}, "
            f"confidence={self.confidence:.2f}, depth={self.depth_level.value})"
        )

    # ── Serialization ─────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "explanation": self.explanation,
            "aliases": self.aliases,
            "parent_ids": self.parent_ids,
            "status": self.status.value,
            "confidence": self.confidence,
            "depth_level": self.depth_level.value,
            "evidence": self.evidence,
            "evidence_quality": self.evidence_quality.value,
            "last_retrieval": self.last_retrieval,
            "stability": self.stability,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Concept":
        # Map legacy status strings
        status = KnowledgeStatus(d.get("status", "unknown"))
        depth = DepthLevel(d.get("depth_level", "exposed"))
        quality = EvidenceQuality(d.get("evidence_quality", "auto"))

        return cls(
            id=d["id"],
            name=d["name"],
            explanation=d["explanation"],
            aliases=d.get("aliases", []),
            parent_ids=d.get("parent_ids", []),
            status=status,
            confidence=d.get("confidence", 0.0),
            depth_level=depth,
            evidence=d.get("evidence", []),
            evidence_quality=quality,
            last_retrieval=d.get("last_retrieval", ""),
            stability=d.get("stability", 0.0),
            last_updated=d.get("last_updated", ""),
        )


class KnowledgeGraph:
    """A graph of concepts representing what a user understands.

    v0.2: O(1) alias lookup via _alias_index, supports forgetting curve decay.
    """

    def __init__(self, concepts: dict[str, Concept] | None = None):
        self.concepts: dict[str, Concept] = concepts or {}
        self._alias_index: dict[str, str] = {}  # alias → concept_id, O(1) lookup
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the alias → concept_id reverse index."""
        self._alias_index = {}
        for c in self.concepts.values():
            self._alias_index[c.id] = c.id
            self._alias_index[c.name.lower()] = c.id
            for alias in c.aliases:
                self._alias_index[alias.lower()] = c.id

    def get(self, id: str) -> Optional[Concept]:
        """Get a concept by id or alias. O(1) via reverse index."""
        if id in self.concepts:
            return self.concepts[id]
        concept_id = self._alias_index.get(id.lower())
        if concept_id:
            return self.concepts.get(concept_id)
        return None

    def is_known(self, id: str) -> bool:
        """Check if a concept is known (also checks parents)."""
        c = self.get(id)
        if c is None:
            return False
        if c.is_known():
            return True
        for pid in c.parent_ids:
            if self.is_known(pid):
                return True
        return False

    def is_learning(self, id: str) -> bool:
        """Check if a concept is in learning status."""
        c = self.get(id)
        if c is None:
            return False
        return c.status == KnowledgeStatus.LEARNING

    def is_unknown(self, id: str) -> bool:
        """Check if a concept is explicitly unknown."""
        c = self.get(id)
        if c is None:
            return True
        if c.is_unknown():
            return True
        if c.parent_ids and all(self.is_unknown(pid) for pid in c.parent_ids):
            return True
        return False

    def needs_explanation(self, id: str) -> bool:
        """Should this concept trigger an explanation?"""
        c = self.get(id)
        if c is None:
            return True
        return c.needs_explanation()

    def add(self, concept: Concept) -> None:
        self.concepts[concept.id] = concept
        self._alias_index[concept.id] = concept.id
        self._alias_index[concept.name.lower()] = concept.id
        for alias in concept.aliases:
            self._alias_index[alias.lower()] = concept.id

    def mark_known(self, id: str, evidence: str = "") -> Optional[Concept]:
        """Mark a concept as known with explicit user evidence."""
        c = self.get(id)
        if c is None:
            return None
        c.status = KnowledgeStatus.KNOWN
        c.confidence = 1.0
        c.evidence_quality = EvidenceQuality.EXPLICIT
        c.depth_level = DepthLevel.COMPREHEND
        c.stability = 0.8
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
        c.evidence_quality = EvidenceQuality.EXPLICIT
        c.stability = 0.3
        if evidence:
            c.evidence.append(evidence)
        c.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return c

    def apply_decay_all(self) -> dict[str, float]:
        """Apply forgetting curve decay to all concepts. Returns updated confidences."""
        now = datetime.now(timezone.utc)
        updates = {}
        for c in self.concepts.values():
            old_conf = c.confidence
            c.apply_decay(now)
            if abs(c.confidence - old_conf) > 0.01:
                updates[c.id] = c.confidence
        return updates

    def get_unknown_for_text(self, text: str) -> list[Concept]:
        """Find concepts mentioned in text that the user doesn't understand."""
        unknown = []
        for c in self.concepts.values():
            if c.needs_explanation():
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
        by_depth = {}
        for c in self.concepts.values():
            d = c.depth_level.value
            by_depth[d] = by_depth.get(d, 0) + 1
        return {
            "total": len(self.concepts),
            "known": known,
            "learning": learning,
            "unknown": unknown,
            "by_depth": by_depth,
            "avg_confidence": round(
                sum(c.confidence for c in self.concepts.values()) /
                max(1, len(self.concepts)), 3
            ),
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
        graph = cls(concepts)
        return graph
