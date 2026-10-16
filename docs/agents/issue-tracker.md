# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues use one file per ticket: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`
- Tickets are numbered from `01` in dependency order
- Triage state is stored in a `Status:` line near the top
- Comments are appended under a `## Comments` heading

## Publishing

When a skill says “publish to the issue tracker”, create the corresponding Markdown file under `.scratch/<feature-slug>/`.

## Fetching tickets

Read the ticket path or issue number supplied by the user.

## Wayfinding

- Map: `.scratch/<effort>/map.md`
- Child ticket: `.scratch/<effort>/issues/<NN>-<slug>.md`
- Each child records `Type:`, `Status:` and `Blocked by:`
- A ticket is ready when all blockers are resolved
- Claim work by changing its status to `claimed`
- Resolve it by adding an `## Answer`, changing its status to `resolved`, and linking the conclusion from `map.md`
