# Domain Docs

This repository uses a single domain context.

## Before exploring

Read these when they exist:

- Root `CONTEXT.md` for project terminology
- Relevant decisions under `docs/adr/`

Missing files are normal. Continue silently; `/domain-modeling` creates them only when real terminology or architectural decisions need recording.

## Layout

```text
/
├── CONTEXT.md
├── docs/adr/
└── app/
```

## Vocabulary

Use terms as defined in `CONTEXT.md` when naming modules, tests, issues and design decisions. If a required term is missing or ambiguous, record it for `/domain-modeling`.

## ADR conflicts

Surface conflicts with existing ADRs explicitly instead of silently replacing the recorded decision.
