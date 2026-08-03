"""Recursive explanation engine — the core loop.

Given a text (e.g., an AI's answer), find concepts the user doesn't understand,
explain them, recursively check those explanations, and return the expanded result.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import KnowledgeStore


def extract_mentioned_concepts(text: str, store: "KnowledgeStore") -> list[str]:
    """Find which concepts from the knowledge graph are mentioned in the text.

    Returns concept IDs sorted by priority (needs-explanation first, known last).
    Learning-status concepts are treated as needing explanation (user hasn't mastered yet).
    """
    text_lower = text.lower()
    mentioned = set()
    for concept in store.graph.concepts.values():
        terms = [concept.name] + concept.aliases
        for term in terms:
            term_lower = term.lower()
            # Use word-boundary regex for all terms to avoid false positives
            # e.g. "rust" should NOT match "trust", "crust", "frustration"
            pattern = r'\b' + re.escape(term_lower) + r'\b'
            try:
                if re.search(pattern, text_lower):
                    mentioned.add(concept.id)
                    break
            except re.error:
                # Fallback: substring match for terms with special chars
                if term_lower in text_lower:
                    mentioned.add(concept.id)
                    break

    # Sort: unknown + learning first (both need explanation), then known
    needs_explanation = [
        id for id in mentioned
        if store.graph.is_unknown(id) or store.graph.is_learning(id)
    ]
    known = [id for id in mentioned if store.graph.is_known(id) and id not in needs_explanation]
    return needs_explanation + known


def explain_recursively(
    text: str,
    store: "KnowledgeStore",
    max_depth: int = 3,
    current_depth: int = 0,
    visited: set[str] | None = None,
) -> str:
    """Recursively expand unknown concepts in text.

    Args:
        text: The text to explain.
        store: Knowledge store to check against.
        max_depth: Maximum recursion depth (prevents infinite loops).
        current_depth: Current recursion depth.
        visited: Set of concept IDs already explained (prevents circular refs).

    Returns:
        Text with unknown concepts explained inline.
    """
    if visited is None:
        visited = set()

    if current_depth >= max_depth:
        return text

    mentioned = extract_mentioned_concepts(text, store)
    needs_explanation = [
        id for id in mentioned
        if (store.graph.is_unknown(id) or store.graph.is_learning(id))
        and id not in visited
    ]

    if not needs_explanation:
        return text

    # Build explanations for concepts needing explanation
    explanations: list[tuple[str, str]] = []
    for concept_id in needs_explanation:
        c = store.graph.get(concept_id)
        if c is None:
            continue
        visited.add(concept_id)
        # Recursively check if the explanation itself has unknowns
        raw_explanation = c.explanation
        expanded = explain_recursively(
            raw_explanation, store, max_depth, current_depth + 1, visited
        )
        explanations.append((c.name, expanded))

    if not explanations:
        return text

    # Attach explanations
    parts = [text, "", "---", "📚 **需要先了解的概念：**", ""]
    for name, expl in explanations:
        parts.append(f"- **{name}**: {expl}")

    return "\n".join(parts)


def check_answer(
    answer: str,
    store: "KnowledgeStore",
    max_depth: int = 3,
) -> dict:
    """Full check: analyze an answer, report what's unknown, optionally expand.

    Returns:
        {
            "text": expanded answer (or original if no unknowns),
            "unknown_concepts": [...],
            "known_concepts_used": [...],
            "expansions_needed": int,
            "max_depth_reached": bool,
        }
    """
    mentioned = extract_mentioned_concepts(answer, store)
    unknown = [id for id in mentioned if store.graph.is_unknown(id) or store.graph.is_learning(id)]
    known = [id for id in mentioned if store.graph.is_known(id)]

    result = {
        "unknown_concepts": [
            {"id": id, "name": store.graph.get(id).name if store.graph.get(id) else id}
            for id in unknown
        ],
        "known_concepts_used": [
            {"id": id, "name": store.graph.get(id).name if store.graph.get(id) else id}
            for id in known
        ],
        "expansions_needed": len(unknown),
        "max_depth_reached": False,
    }

    if unknown and max_depth > 0:
        expanded = explain_recursively(answer, store, max_depth=max_depth)
        result["text"] = expanded
    else:
        result["text"] = answer

    return result
