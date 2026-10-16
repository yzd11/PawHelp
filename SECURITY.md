# Security

## Supported deployment scope

The current release targets a **single trusted local operator**. Bind the main
application and MCP/classifier services to loopback. Do not expose this version
directly to the public Internet or use it with real customer data.

There is no login, registration, RBAC, authenticated customer identity or
multi-tenant isolation. The browser UUID is an anonymous conversation key, not
an identity credential. A separate management menu is not authorization.
Some operations enforce loopback/same-origin access; that is not a complete
security boundary for a public deployment. Reverse proxies are not a substitute
for authentication and ownership checks.

## Data and external models

- Keep `.env`, database dumps, conversation logs, training data and model artifacts
  out of Git. Review Git history as well as the working tree before publishing.
- Knowledge mining, normalization, evaluation and chat may send text to your
  configured model provider. Configure an appropriate provider and obtain the
  necessary consent before using private data. A paid-action confirmation is not
  itself permission to send another person's data.
- Model outputs and retrieved documents are untrusted. Tool arguments require
  validation; business writes require confirmation and ownership checks.
- Demo tools do not execute real payment refunds or connect a human agent.

## Reporting a vulnerability

No public repository or private security contact has been configured yet. Until
one is available, contact the maintainer through the channel that supplied your
copy. Do not publish exploit details, secrets or customer records in public
issues. A private reporting channel must be established before public release.

If a secret leaks, revoke/rotate it first. Adding an ignore rule or deleting the
latest copy does not remove the secret from history. History rewriting requires
explicit coordination with maintainers and existing clones.
