# Contributing to PawHelp

PawHelp is a self-hosted e-commerce assistant. Customer chat, retrieval,
tool-backed workflows and offline knowledge operations are separate modules.
See [project structure](docs/project-structure.md) and [operations](docs/operations.md).

## Development

1. Use Python 3.12 and the committed lockfile. Copy `.env.example` to `.env` and
   configure your own local services. Never commit credentials.
2. Start the existing MySQL/Milvus services as described in the operations guide.
   Verify changes against isolated demo data, not customer data.
3. Run source checks and exercise the behavior affected by your change.
   Include commands and results in your pull request. If a dependency or service
   is unavailable, report the limitation.
4. For browser changes, verify desktop and narrow layouts plus the affected action.
5. Keep demonstrations explicitly labeled. Do not replace missing data with random
   numbers or turn a simulated action into a success claim.

The GitHub workflow runs `uv run pytest` plus offline Python/configuration and
JavaScript syntax checks (`python scripts/check_sources.py`,
`node scripts/check_frontend.cjs`). The suite uses scripted fake models and never
reads your `.env`. Live-MySQL tests are skipped unless `TEST_DATABASE_URL` points to
a disposable database migrated to head; never point it at a database you care about.
Workflow tests share the doubles in `tests/workflow_fakes.py`; extend those instead
of writing another fake model, and do not import one test module from another.
Tests do not replace browser validation for UI changes.

## Changes and compatibility

- Keep business operations outside HTTP handlers. Do not put training on the
  real-time conversation path.
- Preserve database migrations, existing user data and CLI contracts. Document
  breaking changes and migration procedures before implementing them.
- Tests must not call paid models or send customer content to external services.
- Training/evaluation datasets must identify source, split, review status and
  limitations. Do not select thresholds on the held-out test set.
- Supply a reproducible verification for a backend fix and browser evidence for
  UI changes. Future behavioral tests should be small and independently runnable.
  Do not add a framework just to reorganize existing code.

## Pull requests

Explain the user-visible problem, the change, verification and known limitations.
Small, scoped pull requests are easier to maintain. Report security issues using
[SECURITY.md](SECURITY.md), not a public issue containing sensitive details.

By submitting a contribution, you confirm you have permission to contribute it
under the repository's license. Preserve notices for third-party material.
