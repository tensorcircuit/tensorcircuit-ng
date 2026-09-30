"""
ZX-calculus based simulation and optimization

.. warning::

    This module is experimental: its API may change and the whole module may be
    removed at any time without deprecation. Importing it emits
    :py:class:`tensorcircuit.utils.ExperimentalWarning`.
"""

from ..utils import experimental_module_warning
from .stabilizertcircuit import StabilizerTCircuit

experimental_module_warning(__name__)
