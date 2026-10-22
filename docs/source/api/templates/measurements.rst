tensorcircuit.templates.measurements
================================================================================
.. automodule:: tensorcircuit.templates.measurements
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Sparse Heisenberg couplings
--------------------------

``heisenberg_measurements`` avoids XX, YY, or ZZ expectations whose coupling is
a static Python zero, which reduces work for Ising and field-only models.
Tensor-valued XX, YY, and ZZ couplings remain in the computation even when their
value is zero, preserving their derivatives. The existing behavior is retained
when all couplings are zero.
