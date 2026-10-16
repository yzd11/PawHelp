## Agent skills

### Issue tracker

Issues and specs are tracked as local Markdown under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The repo uses the default mattpocock/skills triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo using root `CONTEXT.md` and `docs/adr/`. See `docs/agents/domain.md`.

## Development workflow

### Route by scope

- Clear bugs and small single-file changes: inspect, change, then run one targeted verification.
- Ordinary features: agree on acceptance criteria, then implement directly.
- Ambiguous or cross-module features: use `grill-with-docs`, `to-spec`, `to-tickets`, then `implement`.

Use feature names such as `.scratch/rag-retrieval/`; reserve version numbers for product milestones.

### Keep the codebase lean

Before adding a feature, inspect the touched area for code proven unreachable, superseded, or duplicated. Remove it when the same verification preserves observable behaviour. Keep active public and compatibility contracts.

Existing tests are pruned when their module is touched and they are proven duplicate, obsolete, or coupled only to implementation details.

### Verify at the highest seam

- Test user-observable behaviour at the highest practical seam; a bug normally earns one regression test.
- Validate prompts and data with labelled examples, and browser UI with browser evidence.
- During development, run only tests related to the changed behaviour.
- Run the full suite once at completion when production code, shared interfaces, or dependencies changed.
- When verification fails, report the evidence and stop before another attempt.

### Review and record

- Small changes finish without an independent review.
- Cross-module or high-risk changes receive one code review, with no review loop.
- Use a normal feature branch for larger work; use a worktree only when explicitly requested.
- Commit once per complete task.
- Use the commit and local spec/tickets as the development record. Reserve ADRs for hard-to-reverse, surprising trade-offs.
