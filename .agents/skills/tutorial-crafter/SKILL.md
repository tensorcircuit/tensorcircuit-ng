---
name: tutorial-crafter
description: Transforms a raw TensorCircuit-NG script into a comprehensive, self-contained, and narrative-driven tutorial in Markdown and/or HTML. It acts as an expert technical writer, blending physics/math background, step-by-step code walkthroughs, and HPC programming highlights.
allowed-tools: Bash, Read, Write
---
When tasked with generating a tutorial from a TensorCircuit-NG (TC-NG) script, you act as an Expert Quantum Computing Educator and Technical Writer. Your goal is to produce a self-contained, engaging tutorial that guides the reader from theoretical physics concepts down to the JAX-accelerated code implementation.

### 0. Output Format Selection
- **Default**: Generate both `.md` and `.html` files.
- **Explicit**: If the user specifies a format (e.g., "only MD" or "HTML format"), respect that request.

### 1. Script Analysis & Intent Extraction
- **Understand the Physics**: What is the physical or mathematical goal of the script? (e.g., VQE, QAOA, DMRG).
- **Identify the TC-NG Highlights**: Look for the "TC Way"—such as `tc.backend.vmap`, `jax.jit`, `jax.lax.scan`, or `cotengra` optimizations. Highlight these to educate the user on high-performance practices.

### 2. Tutorial Content Blueprint (Applies to both MD and HTML)
Generate a comprehensive narrative following this structure:
- **Title & Introduction**: Catchy title, 2-3 sentence abstract, and prerequisites.
- **Physics & Mathematical Background**: Theoretical explanation using rigorous LaTeX for all formulas (inline: `$math$`, display: `$$math$$`).
- **Step-by-Step Code Walkthrough**: Break the code into logical snippets with transitional prose explaining the "what" and "why".
- **TC-NG Programming Highlights & Caveats**: Explicitly point out TC-NG/JAX design patterns (e.g., `vmap` vs. loops, `jax.checkpoint`).
- **Results, Visualizations & Conclusion**: Describe expected output and potential next steps.

### 3. HTML Template & Style (Strict Guidelines)
When generating HTML, you MUST use the following premium style and structure to ensure consistency:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>[Tutorial Title]</title>
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <!-- MathJax -->
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <!-- Prism Syntax Highlighting -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <style>
        :root { --primary: #6366f1; --bg-dark: #0f172a; --text-main: #f8fafc; --text-dim: #94a3b8; --highlight: #10b981; }
        body { font-family: 'Inter', sans-serif; background-color: var(--bg-dark); color: var(--text-main); line-height: 1.7; padding: 2rem 1rem; background-image: radial-gradient(circle at top right, rgba(99, 102, 241, 0.1), transparent); background-attachment: fixed; }
        .container { max-width: 900px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 4rem; }
        .logo { width: 400px; margin-bottom: 2rem; }
        h1 { font-family: 'Outfit', sans-serif; font-size: 3rem; background: linear-gradient(135deg, #fff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem; }
        h2 { font-family: 'Outfit', sans-serif; font-size: 2rem; border-bottom: 2px solid rgba(255, 255, 255, 0.05); margin: 2.5rem 0 1.5rem; }
        pre[class*="language-"] { border-radius: 1rem; margin: 1.5rem 0; background: #011627 !important; padding: 1.5rem; }
        blockquote { background: rgba(99, 102, 241, 0.1); border-left: 4px solid var(--primary); padding: 1.25rem 1.5rem; border-radius: 0 1rem 1rem 0; margin: 2rem 0; }
        .highlight-box { background: rgba(16, 185, 129, 0.1); border-left: 4px solid var(--highlight); padding: 1.25rem 1.5rem; border-radius: 0 1rem 1rem 0; margin: 2rem 0; }
        footer { text-align: center; padding: 4rem 0 2rem; border-top: 1px solid rgba(255, 255, 255, 0.1); color: var(--text-dim); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <img src="https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/docs/source/statics/logong.png?raw=true" alt="TensorCircuit-NG Logo" class="logo">
            <h1>[Tutorial Title]</h1>
            <p>[Tagline]</p>
        </header>
        <div class="card"><strong>✨ Abstract:</strong> [Abstract Content]</div>
        <!-- Content Sections here -->
        <footer>Built with <a href="https://github.com/tensorcircuit/tensorcircuit-ng" style="color:var(--primary)">TensorCircuit-NG</a></footer>
    </div>
</body>
</html>
```

### 4. File Saving & Confirmation
- Save `.md` as `[original_name]_tutorial.md` in the same directory as the original script.
- Save `.html` as `[original_name]_tutorial.html` in the same directory as the original script.
- Use the official remote URL for the logo in the <img> tag: https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/docs/source/statics/logong.png?raw=true
- Summarize the completion.
