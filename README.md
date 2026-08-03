# you-know (知道么)

**AI-agnostic personal knowledge model.** Any AI answers you → queries your knowledge graph → checks what you don't understand → recursively explains unknown concepts → returns a response you can actually follow.

## Concept

```
AI wants to answer your question
  → calls check_answer("the answer text")
  → you-know finds unknown concepts
  → recursively expands explanations (up to depth 3)
  → AI delivers the expanded answer
  → you say "I understand X now"
  → AI calls mark_known("X")
  → next time, no explanation needed for X
```

## Architecture

| Component | File | Role |
|-----------|------|------|
| Data models | `src/you_know/models.py` | Concept + KnowledgeGraph types |
| Storage | `src/you_know/store.py` | JSON file persistence + seed data |
| Engine | `src/you_know/engine.py` | Recursive explanation loop |
| MCP Server | `src/you_know/server.py` | Exposes tools to any AI via MCP |

## Install

```bash
cd /Users/ljj/Projects/you-know
pip install -e .
```

## Run as MCP server

```bash
python -m you_know.server
```

## Register with Claude Code

Add to `.claude/settings.json` or `~/.claude/claude.json`:

```json
{
  "mcpServers": {
    "you-know": {
      "command": "python",
      "args": ["-m", "you_know.server"],
      "cwd": "/Users/ljj/Projects/you-know"
    }
  }
}
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `check_answer` | Analyze text, find unknown concepts, recursively expand |
| `lookup_concept` | Query a single concept's status and explanation |
| `mark_known` | Mark a concept as understood |
| `mark_learning` | Mark a concept as being learned |
| `add_concept` | Register a new concept |
| `get_stats` | Graph statistics (total/known/learning/unknown) |
| `list_concepts` | List all concepts, filter by status |

## Data model

```
Concept
  ├── id: kebab-case identifier
  ├── name: human-readable name
  ├── explanation: 1-2 sentence simple explanation
  ├── aliases: alternative names
  ├── parent_ids: broader parent concepts
  ├── status: known | learning | unknown
  ├── confidence: 0.0-1.0
  └── evidence: why we think this status is correct
```

## Project status

v0.1 — MCP server working, JSON file storage, seed data from grok-build learning context.

Next:
- [ ] Hierarchical concept inference (if parent known → child likely known)
- [ ] Batch import from conversation
- [ ] Periodic review of "learning" concepts
