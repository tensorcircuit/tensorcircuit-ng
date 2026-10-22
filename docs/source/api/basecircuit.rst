tensorcircuit.basecircuit
================================================================================
.. automodule:: tensorcircuit.basecircuit
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Batched tensor-network sampling
------------------------------

For qubit circuits, ``sample(batch=..., allow_state=False, format="sample_bin")``
vectorizes the shots before formatting the result. Other explicit output formats
use the same batched computation. Supplying ``status`` with shape
``(batch, nqubits)`` fixes the randomness. Tensor outputs with the default
``jittable=True`` support an outer backend JIT.
The temporary tensor memory can grow with the batch size; use smaller batches
when memory is limited. The default ``format=None`` and qudit sampling retain
their per-shot computation.
