# Permanent development protocol

For every completed development phase, update all three documents in the same
change before reporting completion:

1. `decision.md`: record architectural decisions, reasoning, tradeoffs, and consequences.
2. `README.md` (the repository's readme.md): describe the actual system state,
   configuration, supported commands, and operational limitations.
3. `project_status.md`: list exactly what was built, validation evidence, pending
   work, and the effect on live market analysis and execution.

This rule also applies to fixes, tests, configuration changes, and documentation
phases. When architecture is unchanged, explicitly record that fact in the phase
entry in `decision.md`. Never mark network validation or live readiness complete
on the strength of mocks, compilation, or historical tests. Keep credentials out
of code, documentation, test output, and version control.

The repository is a pure analytical signal engine. Trade lifecycle records are
analytical positions, not evidence of broker orders or fills. Keep network probes
isolated from core engine initialization and order operations.

## Canonical laptop checkout

On this laptop, the user identifies /Volumes/APFS/projects/trading_engine as the
canonical local project root. When working from a Codex workspace, synchronize
completed source, configuration, tests, and documentation changes to that checkout
as requested by the user, preserving existing local edits. Never mirror .env,
credentials, runtime data/logs, or Git metadata. Use the canonical checkout for
commits and pushes after validating the combined changes.
