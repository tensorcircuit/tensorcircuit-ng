Agentic Development
==========================


TensorCircuit-NG is the world's first AI-native quantum programming platform, purpose-built for agentic research and automated scientific discovery. 

.. grid:: 1 1 2 2
   :margin: 4

   .. grid-item-card:: 🚀 Experience Agent-Native Discovery
      :columns: 12 12 6 6
      :link: agent_landing/index.html
      :link-type: url
      :shadow: lg
      :class-card: sd-bg-light sd-text-primary sd-font-weight-bold

      Click here to see how AI agents autonomously solve complex quantum problems in TensorCircuit-NG.

   .. grid-item-card:: Platform Overview
      :columns: 12 12 6 6
      :link: platform/index.html
      :link-type: url
      :shadow: md

      Review the broader TensorCircuit-NG architecture, performance evidence, and adoption landscape.



Why Work Within the Repository?
----------------------------------

To write scripts and applications efficiently with AI coding agents (e.g., ClaudeCode, Cursor, Codex, Antigravity, OpenCode), we strongly recommend working directly within the local repository.


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
*   **sanity-checker**: Systematic audit and refactoring to reduce technical debt, improve abstractions, and ensure codebase health.
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

Initial Prompt for Empty Workspace Setup
----------------------------------------

When starting in a completely empty folder/workspace with an AI agent, you can copy and paste the following initial prompt to bootstrap the agent. 
This instructs the agent to clone the repository, read coding guidelines/skills, and execute tasks accordingly:

.. code-block:: markdown

    You are an AI coding assistant. Please follow these steps to set up our project workspace and solve my requests:
    
    1. Run `git clone https://github.com/tensorcircuit/tensorcircuit-ng.git .` to download the repository in this empty folder.
    2. Install the git-cloned TensorCircuit-NG library locally in the appropriate Python environment. If unsure which Python environment is appropriate, ask the user.
    3. Read `AGENTS.md` at the root of the repository to understand the coding standards.
    4. Review the available agentic workflow scripts and skill files under `.agents/skills/` to see if a specialized workflow is suitable for the current task.
    5. Reference existing implementations in `examples/` and test cases in `tests/` to write differentiable, and JIT-friendly code.
    6. Respond to my later requests, write code, and execute experiments/tests to analyse the results.
    
    Let me know when the initialization is complete and ask for my requests.


AI-Native Documentation
-----------------------

One can also refer to AI-native docs for tensorcircuit-ng: `Devin Deepwiki <https://deepwiki.com/tensorcircuit/tensorcircuit-ng>`_, `Google Code Wiki <https://codewiki.google/github.com/tensorcircuit/tensorcircuit-ng>`_, and `Context7 MCP <https://context7.com/tensorcircuit/tensorcircuit-ng>`_.
