# Skill: Tutorial-Crafter

## Description
The `tutorial-crafter` skill is designed to automatically transform raw TensorCircuit-NG (TC-NG) executable scripts into comprehensive, self-contained, and narrative-driven tutorials in **Markdown** and/or **HTML**. It acts as an Expert Quantum Computing Educator and Technical Writer, blending theoretical physics concepts, step-by-step code walkthroughs, and HPC programming highlights.

## Feature Highlights
- **Script Analysis & Intent Extraction**: Identifies the core mathematical/physical goal (e.g., VQE, DMRG) and extracts TC-NG specific highlights (JAX vectorization, JIT compilation, Cotengra optimizations).
- **Flexible Formats**: Supports dual-output (MD and HTML) with premium, research-journal style styling for HTML exports.
- **Official Branding**: Automatically includes the TensorCircuit-NG logo (`logong.png`) in generated HTML headers.
- **Narrative Content Generation**: Produces a structured tutorial that guides the reader from theory to code implementation with an encouraging and clear style.
- **Rigorous Physics Background**: Provides accessible theoretical explanations with proper LaTeX formatting ($inline$ and $$display$$).
- **Expert Tips & Caveats**: Explicitly points out TC-NG/JAX specific design patterns and potential pitfalls.

## Usage
Command your assistant with:
- `/tutorial-crafter path/to/script.py` (Generates both MD and HTML)
- `/tutorial-crafter path/to/script.py --html` (Generates only HTML)
- `/tutorial-crafter path/to/script.py --md` (Generates only Markdown)

## Tutorial Blueprint
The generated tutorials strictly follow this structure:
1. **Title & Introduction**: Catchy title, abstract, and prerequisites.
2. **Physics & Mathematical Background**: Theoretical explanation with LaTeX formulas.
3. **Step-by-Step Code Walkthrough**: Chunked code snippets with narrative explanations.
4. **TC-NG Programming Highlights & Caveats**: Explicit expert tips and warnings.
5. **Results, Visualizations & Conclusion**: Expected output description, plot explanations, and takeaways.
