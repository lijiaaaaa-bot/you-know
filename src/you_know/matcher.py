"""Fast concept matcher using Aho-Corasick automaton with lemmatization.

Replaces the O(N*M) regex scanning in engine.py with O(T+M) trie traversal.
Includes passive term extraction for auto-growing the concept graph.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import KnowledgeStore


# ── Aho-Corasick wrapper ────────────────────────────────────

class ConceptMatcher:
    """Builds an Aho-Corasick automaton from concept names + aliases.

    Single-pass scan over any text finds all mentioned concepts in O(T+M)
    regardless of how many concepts are in the graph.
    """

    def __init__(self, store: "KnowledgeStore"):
        self.store = store
        self._automaton = None
        self._term_to_id: dict[str, str] = {}  # normalized term → concept_id
        self._dirty = True

    def _ensure_built(self) -> None:
        """(Re)build the automaton if concepts have changed."""
        if not self._dirty and self._automaton is not None:
            return

        try:
            import ahocorasick
            self._automaton = ahocorasick.Automaton()
            use_lib = True
        except ImportError:
            # Pure Python fallback: build a simple trie
            self._trie: dict[str, dict] = {}
            use_lib = False
            self._automaton = None

        self._term_to_id = {}

        if use_lib:
            for concept in self.store.graph.concepts.values():
                terms = self._normalized_terms(concept)
                for term in terms:
                    if term not in self._term_to_id:
                        self._term_to_id[term] = concept.id
                        self._automaton.add_word(term, (concept.id, term))
            self._automaton.make_automaton()
        else:
            for concept in self.store.graph.concepts.values():
                terms = self._normalized_terms(concept)
                for term in terms:
                    if term not in self._term_to_id:
                        self._term_to_id[term] = concept.id
                        self._insert_trie(term, concept.id)

        self._dirty = False

    def _insert_trie(self, term: str, concept_id: str) -> None:
        """Insert a term into the pure-Python trie."""
        node = self._trie
        for ch in term:
            node = node.setdefault(ch, {})
        node["__concept_id__"] = concept_id

    @staticmethod
    def _normalized_terms(concept) -> list[str]:
        """Generate all normalized search terms for a concept."""
        terms = []
        for raw in [concept.name] + concept.aliases:
            t = raw.lower().strip()
            if t:
                terms.append(t)
        return terms

    def find_mentioned(self, text: str) -> list[str]:
        """Return concept IDs mentioned in text, ordered by priority.

        Priority: unknown/learning first, then known.
        """
        self._ensure_built()
        text_lower = text.lower()
        mentioned: set[str] = set()

        if self._automaton is not None:
            # Use pyahocorasick
            for end_idx, (concept_id, term) in self._automaton.iter(text_lower):
                mentioned.add(concept_id)
        else:
            # Pure Python trie walk with word boundaries
            mentioned = self._trie_find_all(text_lower)

        # Sort: needs-explanation first
        store = self.store
        needs = [id for id in mentioned
                 if store.graph.is_unknown(id) or store.graph.is_learning(id)]
        known = [id for id in mentioned
                 if store.graph.is_known(id) and id not in needs]
        return needs + known

    def _trie_find_all(self, text: str) -> set[str]:
        """Pure-Python trie walk: O(T) scan with word-boundary awareness."""
        mentioned: set[str] = set()
        words = re.split(r'(\W+)', text)  # Split keeping delimiters

        # Also try sliding window over the text for multi-word terms
        text_len = len(text)
        for i in range(text_len):
            node = self._trie
            for j in range(i, text_len):
                ch = text[j]
                if ch not in node:
                    break
                node = node[ch]
                cid = node.get("__concept_id__")
                if cid is not None:
                    # Check word boundary
                    prev_char = text[i - 1] if i > 0 else " "
                    next_char = text[j + 1] if j + 1 < text_len else " "
                    if (not prev_char.isalnum() or i == 0 or not text[i - 1].isalnum()) and \
                       (not next_char.isalnum() or j + 1 == text_len or not text[j + 1].isalnum()):
                        mentioned.add(cid)

        return mentioned

    def mark_dirty(self) -> None:
        """Call after adding/removing concepts to force rebuild."""
        self._dirty = True


# ── Lemmatization (lightweight) ──────────────────────────────

# Simple rule-based lemmatizer for common English suffixes.
# Avoids nltk dependency (~40MB data) — covers ~80% of technical terms.

LEMMA_RULES: list[tuple[str, str]] = [
    # Plural → singular
    ("ies", "y"),   # libraries → library
    ("ves", "f"),   # wolves → wolf
    ("ses", "s"),   # classes → class (but also "buses" → "bus" — handled by "es" rule)
    ("es", ""),     # indexes → index, boxes → box
    ("s", ""),      # concepts → concept

    # Verb forms → base
    ("ing", ""),    # running → run (simplified)
    ("ing", "e"),   # making → make
    ("ed", ""),     # matched → match
    ("ed", "e"),    # created → create

    # Adjective → base
    ("er", ""),     # faster → fast
    ("est", ""),    # fastest → fast
    ("ly", ""),     # quickly → quick
    ("tion", "t"),  # reflection → reflect (simplified)
    ("sion", "d"),  # explosion → explosd (approximate)
]

# Common irregular forms relevant to technical vocabulary
IRREGULAR_FORMS: dict[str, str] = {
    "indices": "index",
    "matrices": "matrix",
    "analyses": "analysis",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "children": "child",
    "metadata": "metadata",  # already singular
    "data": "data",          # treated as mass noun
}


def lemmatize(word: str) -> str:
    """Return the base form of a word using rule-based lemmatization.

    Examples:
        lemmatize("reflections") → "reflection"
        lemmatize("indices") → "index"
        lemmatize("running") → "run"
    """
    word_lower = word.lower().strip()
    if len(word_lower) <= 3:
        return word_lower

    # Check irregular forms
    if word_lower in IRREGULAR_FORMS:
        return IRREGULAR_FORMS[word_lower]

    # Try suffix rules
    for suffix, replacement in LEMMA_RULES:
        if word_lower.endswith(suffix) and len(word_lower) - len(suffix) >= 3:
            base = word_lower[:len(word_lower) - len(suffix)] + replacement
            # Handle doubled consonant: "running" → "runn" → "run"
            if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in 'aeiou':
                base = base[:-1]
            if len(base) >= 2:
                return base

    return word_lower


# ── Passive term extraction ──────────────────────────────────

# Patterns for detecting technical terms in English text
TECH_TERM_PATTERNS: list[re.Pattern] = [
    # CamelCase / PascalCase: "BorrowChecker", "TypeScript"
    re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'),

    # snake_case: "borrow_checker"
    re.compile(r'\b[a-z]+(?:_[a-z]+)+\b'),

    # kebab-case: "borrow-checker"
    re.compile(r'\b[a-z]+(?:-[a-z]+)+\b'),

    # ALL_CAPS acronyms: "API", "JSON", "JWT"
    re.compile(r'\b[A-Z]{2,6}\b'),

    # Capitalized single words: "Rust", "Redis", "Kubernetes"
    re.compile(r'\b[A-Z][a-z]{2,}\b'),

    # Dot-separated: "tokio.rs", "actix-web"
    re.compile(r'\b[a-z]+\.[a-z]+(?:\.[a-z]+)*\b'),

    # Versioned terms: "HTTP/2", "ES2024"
    re.compile(r'\b[A-Za-z]+[/-]\d+\b'),
]

# Stop words that look like tech terms but aren't
TERM_STOP_WORDS: set[str] = {
    "the", "and", "for", "that", "with", "this", "from", "have",
    "are", "was", "were", "been", "being", "will", "would", "could",
    "should", "shall", "may", "might", "must", "can", "does", "did",
    "has", "had", "having", "not", "nor", "but", "yet", "still",
    "also", "just", "only", "then", "now", "here", "there", "when",
    "where", "which", "what", "who", "whom", "whose", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "into", "onto", "than",
    "too", "very", "about", "above", "after", "again", "below",
    "between", "during", "under", "while", "these", "those",
    "first", "second", "last", "next", "every", "within", "without",
    "using", "used", "based", "given", "taken", "known", "called",
    "named", "found", "shown", "seen", "made", "said", "done",
    "need", "want", "like", "make", "take", "give", "know", "think",
    "look", "seem", "feel", "become", "happen", "work", "call",
    # Common programming words that aren't concepts
    "the", "this", "that", "these", "those", "then", "else",
    "function", "class", "method", "value", "type", "object",
    "string", "number", "array", "list", "map", "set", "key",
    "data", "code", "file", "line", "test", "case", "result",
    "return", "error", "null", "true", "false", "none", "ok",
    "implement", "implemented", "implementation", "use", "uses",
}


def extract_technical_terms(text: str) -> list[dict]:
    """Extract technical terms from text that could be concept candidates.

    Returns list of {term, context} dicts for terms not in stop words.
    Deduplicated by lowercased form.
    """
    seen: set[str] = set()
    candidates: list[dict] = []

    for pattern in TECH_TERM_PATTERNS:
        for match in pattern.finditer(text):
            term = match.group(0)
            term_lower = term.lower()

            if term_lower in TERM_STOP_WORDS:
                continue
            if len(term) < 2 or len(term) > 60:
                continue
            if term_lower in seen:
                continue

            seen.add(term_lower)

            # Extract surrounding context (±40 chars)
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            context = text[start:end].replace("\n", " ").strip()

            candidates.append({
                "term": term,
                "normalized": term_lower,
                "context": context,
            })

    return candidates


def auto_register_terms(
    text: str,
    store: "KnowledgeStore",
) -> list[str]:
    """Extract terms from text and auto-register unknown ones in the graph.

    Returns list of newly registered concept IDs.
    """
    candidates = extract_technical_terms(text)
    new_ids: list[str] = []

    for c in candidates:
        term = c["term"]
        normalized = c["normalized"]

        # Skip if already in the graph
        if store.lookup(term) is not None:
            continue
        if store.lookup(normalized) is not None:
            continue

        # Create kebab-case ID
        concept_id = re.sub(r'[^a-z0-9]+', '-', normalized).strip('-')
        if not concept_id or len(concept_id) < 2:
            continue

        # Auto-register as unknown
        store.upsert_concept(
            id=concept_id,
            name=term,
            explanation=f"A technical concept encountered in conversation context: \"{c['context']}\"",
            status="unknown",
        )
        new_ids.append(concept_id)

    return new_ids
