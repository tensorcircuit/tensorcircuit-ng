Executable Research Hub
=======================

From published quantum knowledge to executable research infrastructure.

The `Executable Research Hub <reproduce/index.html>`_ is the literature-to-artifact layer of TensorCircuit-NG's agentic research stack. It collects published quantum computing methods that have been translated into runnable, metadata-rich, and independently inspectable TensorCircuit-NG research artifacts.

.. card-carousel:: 1

   .. card:: Open the Reproduce Papers Gallery
      :link: reproduce/index.html
      :shadow: md

      Explore runnable paper reproductions.

What each artifact contains
---------------------------

Each entry is organized around a published paper and documents the target figure or result, implementation script, TensorCircuit-NG APIs, backend, hardware requirements, scaling strategy, and generated output. The gallery is designed for inspection and execution: read the paper, inspect the implementation, and run the artifact yourself.

These are independent reimplementations rather than author-endorsed replications. Many entries are deliberately scaled down for local execution, and each entry records its simplifications and scope. They should be treated as executable illustrations and benchmark artifacts, not as claims that reproduce every detail of the original study.

How it fits the agentic research stack
--------------------------------------

The Hub represents one stage of the broader TensorCircuit-NG research loop:

.. code-block:: text

   Research question -> literature and methods -> implementation -> exploration
   -> optimization -> validation -> reusable research artifact

The `Agentic Quantum Research guide <agentic.html>`_ describes the broader workflows for algorithm discovery, performance optimization, framework translation, code review, and scientific communication.

Contribute an artifact
----------------------

To contribute a new entry, choose a paper, follow the conventions in `examples/reproduce_papers <https://github.com/tensorcircuit/tensorcircuit-ng/tree/master/examples/reproduce_papers>`_, document the reproduction strategy in ``meta.yaml``, generate the declared outputs, and open a pull request.
