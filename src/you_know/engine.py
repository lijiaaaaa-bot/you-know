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

    Returns concept IDs sorted by priority (unknown first).
    """
    mentioned = set()
    for concept in store.graph.concepts.values():
        terms = [concept.name] + concept.aliases
        for term in terms:
            # Match word boundaries for multi-word terms, substring for short ones
            if len(term) > 4:
                pattern = re.escape(term.lower())
                if re.search(pattern, text.lower()):
                    mentioned.add(concept.id)
                    break
            else:
                if term.lower() in text.lower():
                    mentioned.add(concept.id)
                    break

    # Sort: unknown first
    unknown = [id for id in mentioned if store.graph.is_unknown(id)]
    known = [id for id in mentioned if store.graph.is_known(id) and id not in unknown]
    return unknown + known


def explain_recursively(
    text: str,
    store: "KnowledgeStore",
    max_depth: int = 3,
    current_depth: int = 0,
) -> str:
    """Recursively expand unknown concepts in text.

    Args:
        text: The text to explain.
        store: Knowledge store to check against.
        max_depth: Maximum recursion depth (prevents infinite loops).
        current_depth: Current recursion depth.

    Returns:
        Text with unknown concepts explained inline.
    """
    if current_depth >= max_depth:
        return text

    mentioned = extract_mentioned_concepts(text, store)
    unknown = [id for id in mentioned if store.graph.is_unknown(id)]

    if not unknown:
        return text

    # Build explanations for unknown concepts
    explanations: list[tuple[str, str]] = []
    for concept_id in unknown:
        c = store.graph.get(concept_id)
        if c is None:
            continue
        # Recursively check if the explanation itself has unknowns
        raw_explanation = c.explanation
        expanded = explain_recursively(
            raw_explanation, store, max_depth, current_depth + 1
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
    unknown = [id for id in mentioned if store.graph.is_unknown(id)]
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
