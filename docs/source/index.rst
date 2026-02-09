TensorCircuit Next Generation
===========================================================

.. image:: https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/docs/source/statics/logong.png?raw=true
    :target: https://github.com/tensorcircuit/tensorcircuit-ng


**Welcome and congratulations! You have found TensorCircuit: the Next Generation.** 👏 


Introduction
---------------

`TensorCircuit-NG <https://github.com/tensorcircuit/tensorcircuit-ng>`_ is an industrial-grade, open-source high-performance quantum software framework in Python.

It is designed for researchers and engineers who demand **Speed, Flexibility, and Elegance**.

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: 🧬 Built for Humans
      :shadow: md

      Intuitive API that feels like natural quantum physics. Focus on your research, not the boilerplate.

   .. grid-item-card:: 🚀 Extreme Performance
      :shadow: md

      Achieve **10-100x GPU speedup**. Powered by JIT compilation and optimized tensor engines.

   .. grid-item-card:: 🤖 Deep ML Fusion
      :shadow: md

      Native integration with **JAX, TensorFlow, and PyTorch**. Seamlessly combine AD with quantum kernels.

   .. grid-item-card:: ⛓️ Advanced Engines
      :shadow: md

      State-of-the-art support for **Noisy, Analog, Approximate and Stabilizer** simulations.

   .. grid-item-card:: ☁️ Hardware Agnostic
      :shadow: md

      Run your code anywhere: from local CPUs to distributed GPUs/TPUs and **QPU providers** via a unified interface.

   .. grid-item-card:: 🛠 Industrial Strength
      :shadow: md

      Committed to stability and performance. **Long-term support** and proven by extensive research and industrial applications.


.. important::
   Please cite our published `whitepaper <https://quantum-journal.org/papers/q-2023-02-02-912/>`_ when using TensorCircuit or TensorCircuit-NG in your research. The bibtex information is provided by ``tc.cite()``.

.. note::
   The TensorCircuit package is outdated. 
   We recommend upgrading to TensorCircuit-NG for the latest features and improvements. 
   You can upgrade by running the following command:
   ``pip uninstall tensorcircuit && pip install tensorcircuit-ng``


Get Started in Seconds
----------------------

Install with one line:

.. code-block:: bash

    pip install tensorcircuit-ng

Simulate your first circuit:

.. code-block:: python

    import tensorcircuit as tc

    c = tc.Circuit(2)
    c.h(0)
    c.cnot(0, 1)

    print(c.state()) # Ideal Bell state: [0.707, 0, 0, 0.707]
    print(c.expectation_ps(z=[0, 1])) # ZZ expectation: 1.0





Useful Links
--------------------


TensorCircuit is created and now maintained as 
`TensorCircuit-NG <https://github.com/tensorcircuit/tensorcircuit-ng>`_ by `Shi-Xin Zhang <https://github.com/refraction-ray>`_.

The current core authors of TensorCircuit-NG are `Shi-Xin Zhang <https://github.com/refraction-ray>`_ and `Yu-Qin Chen <https://github.com/yutuer21>`_.
We also thank `contributions <https://github.com/tensorcircuit/tensorcircuit-ng/graphs/contributors>`_ from the open source community.

If you have any further questions or collaboration ideas, please use the issue tracker or forum below, or send email to shixinzhang#iphy.ac.cn


.. card-carousel:: 2

   .. card:: Source code
      :link: https://github.com/tensorcircuit/tensorcircuit-ng
      :shadow: md

      GitHub

   
   .. card:: PyPI
      :link:  https://pypi.org/project/tensorcircuit-ng
      :shadow: md

      ``pip install tensorcircuit-ng``


   .. card:: Documentation
      :link: https://tensorcircuit-ng.readthedocs.io
      :shadow: md

      Readthedocs


   .. card:: Whitepaper
      :link: https://quantum-journal.org/papers/q-2023-02-02-912/
      :shadow: md

      *Quantum* journal


   .. card:: Issue Tracker
      :link: https://github.com/tensorcircuit/tensorcircuit-ng/issues
      :shadow: md

      GitHub Issues


   .. card:: Forum
      :link: https://github.com/tensorcircuit/tensorcircuit-ng/discussions
      :shadow: md

      GitHub Discussions



   .. card:: DockerHub
      :link: https://hub.docker.com/repository/docker/tensorcircuit/tensorcircuit
      :shadow: md

      ``docker pull``
      

   .. card:: Application
      :link: https://github.com/tensorcircuit/tensorcircuit-ng#research-and-applications
      :shadow: md

      Research using TC





Unified Quantum Programming
------------------------------

TensorCircuit-NG is building the future of unified quantum computing infrastructures.

.. grid:: 1 2 4 4
   :margin: 0
   :padding: 0
   :gutter: 3

   .. grid-item-card:: 🛠 Unified Backends
      :columns: 12 6 3 3
      :shadow: md

      JAX, TensorFlow, PyTorch, Numpy, Cupy

   .. grid-item-card:: 💻 Unified Devices
      :columns: 12 6 3 3
      :shadow: md

      CPU, GPU, and TPU support

   .. grid-item-card:: 🏛 Unified Providers
      :columns: 12 6 3 3
      :shadow: md

      QPUs from major vendors

   .. grid-item-card:: 🌐 Unified Resources
      :columns: 12 6 3 3
      :shadow: md

      Local, Cloud, and HPC environments


.. grid:: 1 2 4 4
   :margin: 0
   :padding: 0
   :gutter: 3

   .. grid-item-card:: 🎛 Unified Interfaces
      :columns: 12 6 3 3
      :shadow: md

      Numerical sim and hardware experiments

   .. grid-item-card:: ⚙️ Unified Engines
      :columns: 12 6 3 3
      :shadow: md

      Ideal, Noisy, Analog, and Stabilizer

   .. grid-item-card:: 📝 Unified Representations
      :columns: 12 6 3 3
      :shadow: md

      Qiskit, OpenQASM, Cirq, and IR

   .. grid-item-card:: 🧊 Unified Objects
      :columns: 12 6 3 3
      :shadow: md

      Neural Nets, Tensor Nets, and Circuits




Reference Documentation
----------------------------

The following documentation sections briefly introduce TensorCircuit-NG to the users and developpers.

.. toctree::
   :maxdepth: 2

   quickstart.rst
   advance.rst
   faq.rst
   sharpbits.rst
   infras.rst
   contribution.rst

Tutorials
---------------------

The following documentation sections include integrated examples in the form of Jupyter Notebook.

.. toctree-filt::
   :maxdepth: 2

   :zh:tutorial.rst
   :zh:whitepapertoc.rst
   :en:tutorial_cn.rst
   :en:whitepapertoc_cn.rst
   :en:textbooktoc.rst



API References
=======================

.. toctree::
   :maxdepth: 2
    
   modules.rst
    

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Join the Community
==================

We are always looking for contributors and collaborators!

*   Check out our `GitHub Discussions <https://github.com/tensorcircuit/tensorcircuit-ng/discussions>`_ for questions and show-and-tell.
*   Report bugs or request features via `GitHub Issues <https://github.com/tensorcircuit/tensorcircuit-ng/issues>`_.
*   Star the repository to stay updated! ⭐
