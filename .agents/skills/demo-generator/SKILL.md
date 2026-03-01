---
name: demo-generator
description: Transforms a TensorCircuit-NG script into an interactive, sleek, and high-performance GUI application using Streamlit. It intelligently extracts the most impressive aspects of the physics simulation and presents them via interactive widgets and real-time visualizations.
allowed-tools: Bash, Read, Write
---

When tasked with creating a Streamlit demo from a TensorCircuit-NG (TC-NG) script, you act as a Creative Full-Stack Quantum Scientist. Your goal is to build an app that doesn't just "run the script," but makes the underlying physics alive, interactive, and **surprising** for an audience. Use your imagination to create a 'wow' effect.

### 0. App Metadata & Docstring
The generated `[original_name]_app.py` MUST begin with a standardized docstring:
```python
"""
Streamlit Interactive Demo: [App Title]
Origin: Based on [path/to/original_script.py]
Usage: streamlit run [original_name]_app.py
Description: [Brief 1-sentence description of the interactivity and goal]
"""
```

### 1. Intent & Interactivity Extraction
- **Identify Controllable Parameters**: What variables in the script are "fun" to change? (e.g., lattice size $L$, bias $\theta$, noise rate $p$, number of qubits $N$).
- **Identify Visual "Hooks"**: What is the most impressive result? (e.g., a 3D phase transition 'volcano', a real-time circuit animation, or a statevector magnitude heatmap).
- **Be Imaginative & Surprising**: Don't just settle for standard line plots. If the physics allows, create 3D surfaces, interactive phase diagrams, or evolving probability landscapes that make the user say "Wow."
- **Plan the Story**: How should the user interact with it? (e.g., "Adjust the temperature and watch the magnetization landscape buckle").

### 2. Standardized App Structure
The generated `[original_name]_app.py` should follow this professional blueprint:

#### A. Global Styling & Branding
- Use `st.set_page_config` with a wide layout and a custom title.
- Integrate the official TensorCircuit-NG logo from: `https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/docs/source/statics/logong.png?raw=true`.
- Apply custom CSS for a premium "Dark/Glassmorphism" look if possible.

#### B. Sidebar Controls
- Put all simulation parameters in the `st.sidebar`.
- Use `st.sidebar.slider`, `num_input`, or `selectbox`.
- Use clear labels and helpful tooltips for each parameter.

#### C. Main Area: Theoretical Context
- Use `st.title` and `st.markdown`.
- Explicitly explain the physics intent of the demo using LaTeX math.
- **Dynamic Problem Preview**: Before the simulation starts, DO NOT use stock photos or irrelevant external images. Instead, generate a static plot that previews the current configuration (e.g., draw the $n \times m$ lattice grid using NetworkX/Matplotlib, or use `tc.visualize` to show the target circuit structure). **Never use distracting generic stock photos from Unsplash etc.**

#### D. Main Area: The Interactive Simulation
- **Caching is Critical**: Use `@st.cache_data` or `@st.cache_resource` for expensive TC-NG simulations to ensure a smooth, lag-free UI.
- **Progressive Disclosure**: Show "Running Simulation..." indicators while the JIT is compiling or the contractor is working.
- **Dynamic Visuals**: Use `st.pyplot(fig)`, `st.plotly_chart(fig)`, or `st.altair_chart(fig)` for plots.
- **Real-time Metrics**: Use `st.columns` and `st.metric` to display final values (e.g., Energy, Fidelity).

### 3. Implementation Best Practices
- **Standard Imports**: Always import `streamlit as st`, `tensorcircuit as tc`, and `jax.numpy as jnp`.
- **Backend Setup**: Explicitly `tc.set_backend("jax")` and handle dtypes correctly.
- **Error Handling**: Wrap the simulation in try-except blocks to catch potential OOM or out-of-bounds parameters, and display friendly warnings in the UI using `st.error`.

### 4. Output & Delivery
- Save the script as `[original_name]_app.py` in the same directory as the original script.
- Provide the exact terminal command to run the app: `streamlit run [original_name]_app.py`.
- Summarize the interactive features you've added and why they make the demo impressive.
