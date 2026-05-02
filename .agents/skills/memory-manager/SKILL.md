---
name: memory-manager
description: Maintain TensorCircuit-NG development memory in `.agents/memory/`. Use `update` mode to save durable lessons from the current session into the right memory file, and use `dream` mode to scan and refactor the whole memory set for lower redundancy, clearer taxonomy, and better long-term usefulness.
allowed-tools: Bash, Read, Grep, Glob, Write
---

When activated, you act as a concise curator for TensorCircuit-NG's development memory. The memory is for future AI development guidance, not for journaling past agent activity.

## Core rules

- Start from `.agents/memory/index.md`. Treat it as the authority for the memory taxonomy and the progressive-disclosure entrypoint.
- Keep memory generic, durable, and repo-relevant. Store lessons that future agents are likely to need again.
- Do not store local or private facts such as conda environment names, usernames, hostnames, absolute paths, branch names, timestamps, temporary files, or "AI did X" narratives.
- Prefer protocols, invariants, and recurring pitfalls over anecdotes. Rewrite one-off debugging stories into short reusable guidance.
- Keep edits concise. Avoid long examples unless the detail is essential to avoid future mistakes.
- Update an existing memory file when possible. Create a new one only when the lesson does not fit the current taxonomy, then add it to `index.md`.

## Mode selection

- The user may explicitly select a mode when invoking the skill, for example `$memory-manager update` or `$memory-manager dream`. Treat that explicit choice as authoritative.
- Use `update` mode when the goal is to save what was learned in the current task or session.
- Use `dream` mode when the goal is to clean up the whole memory set by scanning, merging, renaming, and refactoring memory files.
- If the user does not specify a mode explicitly, infer it from the request. Incremental capture implies `update`; full-memory cleanup implies `dream`.

## Update mode

Use this mode after meaningful work has already happened and there may be durable lessons worth preserving.

### Workflow

1. Read `.agents/memory/index.md`, then only the memory file or files closest to the current task.
2. Inspect the current session evidence: the user request, relevant diffs, tests, failures, fixes, and non-obvious conclusions.
3. Extract only lessons that are:
   - durable across future work,
   - specific enough to be useful,
   - generic enough to be public and reusable.
4. Discard anything that is obvious from the code, too local to the machine/session, or too speculative.
5. Write the lesson into the best existing memory file as short bullets or short sections. If a new file is necessary, create it with a stable topic name and add it to `index.md`.
6. Keep the resulting memory entry future-facing. Explain what to do, what to avoid, or what invariant matters.

### What usually belongs

- Backend quirks that are easy to miss.
- AD, JIT, tracing, or contraction behaviors that cause recurring bugs.
- API invariants, reconstruction rules, serialization constraints, and testing protocols.
- Performance patterns that materially change algorithmic behavior or memory usage.

### What usually does not belong

- Session summaries, timelines, or status reports.
- Local environment setup facts.
- Temporary workarounds without a verified root cause.
- Trivial reminders that any agent can infer from reading the touched file.

## Dream mode

Use this mode to improve the quality of the memory system itself.

### Workflow

1. Read `.agents/memory/index.md`, then scan every memory file referenced by it.
2. Identify overlap, fragmented topics, stale naming, excessive specialization, and high-entropy notes that should be generalized or removed.
3. Merge related files when their boundary is too fine-grained. Split only if a file has become hard to scan or no longer supports progressive disclosure.
4. Rewrite notes into concise, referenceable guidance. Prefer stable topic headings and short bullets over long narratives.
5. Remove redundancy, repetitive wording, and content that merely repeats `AGENTS.md` unless the memory adds a non-obvious repo-specific refinement.
6. Update `index.md` so it remains a clean summary of the current memory map.

### Dream-mode quality bar

- Each file should have a clear topic boundary.
- The index should let an agent choose the right file quickly.
- The memory should stay small enough to scan but rich enough to prevent repeated rediscovery.
- Notes should be informative for future TensorCircuit-NG work, not a log of past maintenance.

## Output expectations

- In `update` mode, make the smallest edit that captures the durable lesson well.
- In `dream` mode, optimize for a better overall memory system, even if that means renaming, merging, or deleting memory files.
- After editing, summarize what changed, what was intentionally excluded, and any remaining taxonomy gaps.
