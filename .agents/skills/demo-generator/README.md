# Skill: Demo-Generator

## Description
The `demo-generator` skill transforms raw TensorCircuit-NG scripts into interactive, polished, and high-performance GUI applications using **Streamlit**. Instead of a static translation, this skill identifies the core narrative and "physics hooks" of a script to create a compelling, interactive showcase of quantum simulations.

## Key Features
- **Interactive Widgets**: Automatically maps controllable parameters (qubits, noise, angles) to sidebar sliders and inputs.
- **High-Performance Caching**: Leverages Streamlit's caching mechanisms alongside JAX's JIT to provide a smooth, real-time user experience even for complex simulations.
- **Premium Visualization**: Focuses on presenting results through clean, professional plots and dynamic metrics.
- **Storytelling approach**: Organizes the app with theoretical context, interactive controls, and visual results to guide the user through the physics.

## When to Use
Use this skill when you want to:
- Turn a research script into a presentation tool.
- Create an interactive playground for exploring how parameters affect a quantum algorithm (e.g., watching a VQE converge).
- Provide a non-coding user with a way to interact with your simulations.
- Showcase the extreme speed and flexibility of TensorCircuit-NG through real-time feedback.

## Usage
Simply point the assistant to your script:
`@[/demo-generator] examples/your_script.py`

The assistant will generate an `app_your_script.py` file which you can run via:
`streamlit run app_your_script.py`
