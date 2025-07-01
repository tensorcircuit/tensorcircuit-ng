TensorCircuit Next Generation
===========================================================

.. image:: https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/docs/source/statics/logong.png?raw=true
    :target: https://github.com/tensorcircuit/tensorcircuit-ng


**Welcome and congratulations! You have found TensorCircuit: the Next Generation.** 👏 


Introduction
---------------

`TensorCircuit-NG <https://github.com/tensorcircuit/tensorcircuit-ng>`_ is an open-source high-performance quantum software framework in Python.

* It is built for humans. 👽

* It is designed for speed, flexibility and elegance. 🚀

* It is empowered by advanced tensor network simulator engines. 🔋

* It is ready for quantum hardware access with CPU/GPU/QPU (local/cloud) hybrid solutions. 🖥

* It is implemented with industry-standard machine learning frameworks: TensorFlow, JAX, and PyTorch. 🤖

* It is flexible and powerful to build and simulate tensor networks, neural networks and quantum circuits together. 🧠

* It is compatible with machine learning engineering paradigms: automatic differentiation, just-in-time compilation, vectorized parallelism and GPU acceleration. 🛠

With the help of TensorCircuit-NG, now get ready to efficiently and elegantly solve interesting and challenging quantum computing and quantum many-body problems: from academic research prototype to industry application deployment.

.. important::
   Please cite the `whitepaper <https://quantum-journal.org/papers/q-2023-02-02-912/>`_ when using TensorCircuit or TensorCircuit-NG in your research. The bibtex information is provided by ``tc.cite()``.

.. note::
   The TensorCircuit package is outdated. 
   We recommend upgrading to TensorCircuit-NG for the latest features and improvements. 
   You can upgrade by running the following command:
   ``pip uninstall tensorcircuit && pip install tensorcircuit-ng``


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

TensorCircuit-NG is unifying infrastructures and interfaces for quantum computing.

.. grid:: 1 2 4 4
   :margin: 0
   :padding: 0
   :gutter: 2

   .. grid-item-card:: Unified Backends
      :columns: 12 6 3 3
      :shadow: md

      Jax/TensorFlow/PyTorch/Numpy/Cupy

   .. grid-item-card:: Unified Devices
      :columns: 12 6 3 3
      :shadow: md

      CPU/GPU/TPU

   .. grid-item-card:: Unified Providers
      :columns: 12 6 3 3
      :shadow: md

      QPUs from different vendors

   .. grid-item-card:: Unified Resources
      :columns: 12 6 3 3
      :shadow: md

      local/cloud/HPC


.. grid:: 1 2 4 4
   :margin: 0
   :padding: 0
   :gutter: 2

   .. grid-item-card:: Unified Interfaces
      :columns: 12 6 3 3
      :shadow: md

      numerical sim/hardware exp

   .. grid-item-card:: Unified Engines
      :columns: 12 6 3 3
      :shadow: md

      ideal/noisy/approx/analog/stabilizer

   .. grid-item-card:: Unified Representations
      :columns: 12 6 3 3
      :shadow: md

      from/to_IR/qiskit/openqasm/json

   .. grid-item-card:: Unified Objects
      :columns: 12 6 3 3
      :shadow: md

      neural-net/tensor-net/quantum-circuit




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
