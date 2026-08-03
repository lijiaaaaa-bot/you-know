"""Recursive explanation engine — the core loop.

v0.2 — Enhanced with:
  - Aho-Corasick fast matching via ConceptMatcher
  - Passive term auto-registration
  - Breadth-first explanation (cognitive-load-aware)
  - Context-aware max_depth
  - BKT-style confidence inference from conversation
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .matcher import ConceptMatcher, auto_register_terms, lemmatize
from .models import DepthLevel, EvidenceQuality, KnowledgeStatus

if TYPE_CHECKING:
    from .store import KnowledgeStore


# ── Mention extraction (delegates to matcher) ─────────────

def extract_mentioned_concepts(
    text: str,
    store: "KnowledgeStore",
    matcher: ConceptMatcher | None = None,
) -> list[str]:
    """Find which concepts from the knowledge graph are mentioned in the text.

    Uses Aho-Corasick automaton for O(T+M) matching.
    Returns concept IDs sorted by priority (needs-explanation first).
    """
    if matcher is None:
        matcher = ConceptMatcher(store)
    return matcher.find_mentioned(text)


# ── Breadth-first explanation ─────────────────────────────

def explain_breadth_first(
    text: str,
    store: "KnowledgeStore",
    max_depth: int = 2,
    matcher: ConceptMatcher | None = None,
) -> str:
    """Expand unknown concepts breadth-first rather than depth-first.

    Cognitive science basis (Cowan 2001, Miller 1956):
      - Working memory limited to 4±1 chunks
      - Depth 3 already exceeds this for novices
      - Breadth-first lets user manage cognitive load

    Instead of recursively drilling down, we:
      1. Show all level-1 concepts needing explanation
      2. For each, show ONE level of sub-concepts
      3. Stop at depth 2 (cognitively safe default)
    """
    if matcher is None:
        matcher = ConceptMatcher(store)

    # Level 1: find all concepts in the original text that need explanation
    mentioned = matcher.find_mentioned(text)
    level1_ids = [
        id for id in mentioned
        if store.graph.needs_explanation(id)
    ]

    if not level1_ids:
        return text

    explanations: list[str] = []

    for concept_id in level1_ids:
        c = store.graph.get(concept_id)
        if c is None:
            continue

        explanation = c.explanation

        # Level 2: check if the explanation itself has unknowns
        if max_depth >= 2:
            sub_mentioned = matcher.find_mentioned(explanation)
            sub_unknowns = [
                id for id in sub_mentioned
                if store.graph.needs_explanation(id)
                and id != concept_id  # avoid self-reference
            ]

            if sub_unknowns:
                sub_parts = []
                for sub_id in sub_unknowns[:3]:  # Limit sub-concepts to avoid overload
                    sub_c = store.graph.get(sub_id)
                    if sub_c:
                        sub_parts.append(f"    - **{sub_c.name}**: {sub_c.explanation}")
                if sub_parts:
                    explanation += "\n" + "\n".join(sub_parts)

        explanations.append(f"- **{c.name}** ({c.depth_level.value}): {explanation}")

    if not explanations:
        return text

    parts = [text, "", "---", "📚 **需要先了解的概念：**", ""]
    parts.extend(explanations)

    return "\n".join(parts)


# ── Context-aware max_depth ────────────────────────────────

def compute_max_depth(
    text: str,
    concepts: list[str],
    store: "KnowledgeStore",
) -> int:
    """Dynamically compute the appropriate explanation depth.

    Based on:
      - Number of unknown concepts (more → shallower to avoid overload)
      - Concept depth levels (deeper existing knowledge → can go deeper)
      - Text complexity (longer text → user has more context)

    Returns depth in [1, 3].
    """
    unknown_count = len([
        id for id in concepts
        if store.graph.needs_explanation(id)
    ])

    # Cognitive load: too many unknowns → keep it shallow
    if unknown_count >= 5:
        return 1
    elif unknown_count >= 3:
        return 2
    else:
        return 3


# ── Full check_answer (backward-compatible) ────────────────

def check_answer(
    answer: str,
    store: "KnowledgeStore",
    max_depth: int | None = None,
    matcher: ConceptMatcher | None = None,
) -> dict:
    """Full check: analyze an answer, report what's unknown, expand if needed.

    v0.2: Uses breadth-first explanation with context-aware depth.
    Also auto-registers previously unseen technical terms.

    Returns:
        {
            "text": expanded answer (or original if no unknowns),
            "unknown_concepts": [...],
            "known_concepts_used": [...],
            "expansions_needed": int,
            "max_depth_reached": bool,
            "new_concepts_registered": [...],
        }
    """
    if matcher is None:
        matcher = ConceptMatcher(store)

    mentioned = matcher.find_mentioned(answer)

    unknown = [
        id for id in mentioned
        if store.graph.needs_explanation(id)
    ]
    known = [
        id for id in mentioned
        if store.graph.is_known(id)
    ]

    # Auto-register unseen technical terms (passive growth)
    new_ids = auto_register_terms(answer, store)
    if new_ids:
        matcher.mark_dirty()

    # Compute appropriate depth
    if max_depth is None:
        max_depth = compute_max_depth(answer, unknown, store)

    result = {
        "unknown_concepts": [
            {
                "id": id,
                "name": store.graph.get(id).name if store.graph.get(id) else id,
                "depth_level": (
                    store.graph.get(id).depth_level.value
                    if store.graph.get(id) else "exposed"
                ),
            }
            for id in unknown
        ],
        "known_concepts_used": [
            {
                "id": id,
                "name": store.graph.get(id).name if store.graph.get(id) else id,
            }
            for id in known
        ],
        "expansions_needed": len(unknown),
        "max_depth_used": max_depth,
        "new_concepts_registered": new_ids,
    }

    if unknown and max_depth > 0:
        expanded = explain_breadth_first(answer, store, max_depth=max_depth, matcher=matcher)
        result["text"] = expanded
    else:
        result["text"] = answer

    return result


# ── Passive inference from conversation ────────────────────

def analyze_conversation(
    user_message: str,
    ai_response: str,
    store: "KnowledgeStore",
    matcher: ConceptMatcher | None = None,
) -> dict:
    """Analyze a conversation turn to passively infer concept understanding.

    Layer 1 heuristics (fast, rule-based):
      - User mentions a concept name → exposure evidence
      - User uses explanatory language ("because", "so", "that is") → comprehension
      - User asks a deep question ("why", "how does") → engagement signal
      - User asks a definition ("what is") → unknown signal

    Returns inference results that the caller can apply via update_confidence.
    """
    if matcher is None:
        matcher = ConceptMatcher(store)

    user_lower = user_message.lower()
    inferences: list[dict] = []

    # Find concepts mentioned by the user
    mentioned = matcher.find_mentioned(user_message)

    for concept_id in mentioned:
        c = store.graph.get(concept_id)
        if c is None:
            continue

        signal = 0.0
        quality = EvidenceQuality.HEURISTIC
        reasoning = []

        # Heuristic 1: User uses the concept name → at least exposure
        reasoning.append("user mentioned the concept")

        # Heuristic 2: Explanatory language → comprehension signal
        explanatory_markers = [
            "因为", "所以", "就是", "意思是", "也就是说",
            "because", "so", "meaning", "that is", "in other words",
            "therefore", "thus", "hence", "basically",
        ]
        if any(m in user_lower for m in explanatory_markers):
            signal += 0.3
            quality = EvidenceQuality.BEHAVIORAL
            reasoning.append("used explanatory language")

        # Heuristic 3: Deep/analytical question → strong engagement
        deep_markers = [
            "为什么", "怎么会", "原理", "底层",
            "why", "how does", "how do", "internally",
            "mechanism", "under the hood",
        ]
        if any(m in user_lower for m in deep_markers):
            signal += 0.4
            quality = EvidenceQuality.BEHAVIORAL
            reasoning.append("asked deep/analytical question")

        # Heuristic 4: Definition question → unknown signal
        def_markers = [
            "什么是", "什么意思", "怎么理解",
            "what is", "what does", "define", "definition",
            "explain", "tell me about",
        ]
        if any(m in user_lower for m in def_markers):
            signal -= 0.3
            reasoning.append("asked for definition (unknown signal)")

        # Heuristic 5: Correct usage in code/command → apply signal
        code_markers = ["```", "fn ", "def ", "class ", "import ", "const ", "let "]
        if any(m in user_message for m in code_markers):
            signal += 0.2
            reasoning.append("used in code context")

        # Apply inference only if signal is meaningful
        if abs(signal) >= 0.2:
            old_conf = c.confidence
            c.update_confidence(signal, quality)
            inferences.append({
                "concept_id": concept_id,
                "concept_name": c.name,
                "signal": round(signal, 2),
                "quality": quality.value,
                "confidence_before": round(old_conf, 3),
                "confidence_after": round(c.confidence, 3),
                "reasoning": reasoning,
            })

        # Record retrieval for forgetting curve
        c.record_retrieval()

    # Also auto-register terms from the AI response (passive growth)
    new_ids = auto_register_terms(ai_response, store)
    if new_ids:
        matcher.mark_dirty()

    return {
        "inferences": inferences,
        "inferences_count": len(inferences),
        "new_concepts_registered": new_ids,
    }


def get_learning_summary(store: "KnowledgeStore") -> dict:
    """Generate a learning dashboard summary.

    For Pull mode — user can call this to see their knowledge state.
    """
    # Apply decay first to get current state
    decayed = store.graph.apply_decay_all()

    stats = store.graph.stats()

    # Top concepts by confidence (strongest knowledge)
    by_conf = sorted(
        store.graph.concepts.values(),
        key=lambda c: c.confidence, reverse=True
    )
    strongest = [
        {"name": c.name, "confidence": round(c.confidence, 2), "depth": c.depth_level.value}
        for c in by_conf[:5]
        if c.confidence > 0.5
    ]

    # Concepts needing attention (learning + low confidence)
    needs_attention = [
        {"name": c.name, "confidence": round(c.confidence, 2),
         "depth": c.depth_level.value, "status": c.status.value}
        for c in store.graph.concepts.values()
        if c.needs_explanation() and c.confidence > 0
    ]
    needs_attention.sort(key=lambda x: x["confidence"])

    # Recently decayed
    recently_decayed = [
        {"concept_id": id, "new_confidence": round(conf, 3)}
        for id, conf in decayed.items()
    ][:5]

    # Blind spots: concepts in graph that user has never engaged with
    blind_spots = [
        {"name": c.name, "confidence": 0.0}
        for c in store.graph.concepts.values()
        if c.confidence == 0.0 and c.status == KnowledgeStatus.UNKNOWN
    ][:5]

    return {
        "stats": stats,
        "strongest_concepts": strongest,
        "needs_attention": needs_attention[:10],
        "recently_decayed": recently_decayed,
        "blind_spots": blind_spots,
        "recommendation": (
            f"You have {stats['unknown']} unknown concepts. "
            f"Focus on the {len(needs_attention)} concepts in 'needs_attention'."
        ) if needs_attention else "Your knowledge graph is in good shape!",
    }
