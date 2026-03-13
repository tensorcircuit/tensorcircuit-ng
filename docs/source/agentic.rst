Agentic Development
==========================


TensorCircuit-NG is the world's first AI-native quantum programming platform, purpose-built for agentic research and automated scientific discovery. 

.. grid:: 1
   :margin: 4

   .. grid-item-card:: 🚀 Experience Agent-Native Discovery
      :link: agent_landing/index.html
      :link-type: url
      :shadow: lg
      :class-card: sd-bg-light sd-text-primary sd-font-weight-bold

      Click here to see how AI agents autonomously solve complex quantum problems in TensorCircuit-NG.



Why Work Within the Repository?
----------------------------------

To write scripts and applications efficiently with AI coding agents (e.g., ClaudeCode, Cursor, Codex, Antigravity, Gemini-CLI, OpenCode), we strongly recommend working directly within the local repository.


1. **Rich Context**: The repository contains over 100 scripts in ``examples/`` and extensive test cases in ``tests/``. These provide essential references that significantly reduce AI hallucinations and help the agent understand idiomatic usage.
2. **Built-in Rules**: We provide a dedicated `AGENTS.md <https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/AGENTS.md>`_ file. It serves as the "handbook" (similar to ``CLAUDE.md``) for AI agents, defining coding standards and best practices.
3. **Specialized Agentic Skills**: The ``.agents/skills/`` directory contains specialized workflows to guide AI assistants on complex, multi-step tasks.

Specialized Agentic Skills
--------------------------

TensorCircuit-NG includes built-in agentic skills that can be activated by compatible AI agents to perform advanced tasks:

*   **arxiv-reproduce**: Autonomously reproduces arXiv papers with standardized output and code quality validation.
*   **performance-optimize**: Scientific execution and memory optimization workflow (JAX scanning, vectorized parallelism, etc.).
*   **tc-rosetta**: End-to-end framework translation (from Qiskit, PennyLane, etc.) with intrinsic mathematical intent rewriting.
*   **tutorial-crafter**: Transforms raw scripts into comprehensive, narrative-driven educational tutorials.
*   **demo-generator**: Transforms scripts into interactive, high-performance Streamlit GUI applications.
*   **code-reviewer**: Autonomously reviews and refactors code for mathematical correctness and performance.
*   **meta-explorer**: High-intensity autonomous research agent for circuit architecture and optimization strategy discovery (VQE, QML, QAOA, etc.).

Recommended Workflow
--------------------

1. **Clone the repository**: 
   
   .. code-block:: bash

      git clone https://github.com/tensorcircuit/tensorcircuit-ng.git

2. **Switch to a local playground branch**: 
   
   .. code-block:: bash

      git checkout -b my-playground

3. **Open the repository folder in your AI IDE**: Start writing TC-NG-based scripts using natural language instructions.

By integrating extreme performance with an autonomous, intent-driven AI workflow, TensorCircuit-NG empowers researchers to transition from manual coding to automated scientific discovery.
