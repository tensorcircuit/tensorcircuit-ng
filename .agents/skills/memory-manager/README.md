# Skill: Memory Manager

## Description
The `memory-manager` skill maintains TensorCircuit-NG's development memory in `.agents/memory/`. It keeps the memory concise, generic, and useful for future AI development rather than recording session history.

## Modes
- **Update**: Extract durable lessons and protocols from the current session and save them in the most appropriate memory file.
- **Dream**: Scan the full memory set, merge overlapping notes, remove redundancy, and improve the overall taxonomy and progressive-disclosure structure.

## Memory Rules
- Start from `.agents/memory/index.md`.
- Prefer stable protocols, invariants, and recurring pitfalls over anecdotes.
- Do not store local or private machine details such as environment names, paths, usernames, or timestamps.
- Keep notes short, referenceable, and future-helpful.

## When to Use
Use this skill when:
- a task produced non-obvious lessons worth preserving for future work;
- the memory files have become fragmented, repetitive, or hard to navigate;
- you want to improve the memory structure without turning it into a maintenance log.
