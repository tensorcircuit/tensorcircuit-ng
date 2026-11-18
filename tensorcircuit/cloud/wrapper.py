"""
higher level wrapper shortcut for submit_task
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import logging
import time

import numpy as np

from ..circuit import Circuit
from ..results import counts
from ..results.readout_mitigation import ReadoutMit
from ..utils import is_sequence
from ..cons import backend
from ..quantum import ps2xyz
from ..compiler import DefaultCompiler
from ..compiler.simple_compiler import simple_compile
from ..symbolcircuit import SymbolCircuit
from .apis import submit_task, get_device
from .abstraction import Device

logger = logging.getLogger(__name__)
Tensor = Any


def batch_submit_template(
    device: str, batch_limit: int = 64, **kws: Any
) -> Callable[..., List[counts.ct]]:
    def run(
        cs: Union[Circuit, Sequence[Circuit]], shots: int = 8192, **nkws: Any
    ) -> List[counts.ct]:
        """
        batch circuit running alternative
        """
        single = False
        if not is_sequence(cs):
            cs = [cs]  # type: ignore
            single = True
        kws.update(nkws)
        if len(cs) <= batch_limit:  # type: ignore
            css = [cs]
        else:
            ntimes = len(cs) // batch_limit  # type: ignore
            if ntimes * batch_limit == len(cs):  # type: ignore
                css = [
                    cs[i * batch_limit : (i + 1) * batch_limit] for i in range(ntimes)  # type: ignore
                ]
            else:
                css = [
                    cs[i * batch_limit : (i + 1) * batch_limit] for i in range(ntimes)  # type: ignore
                ] + [
                    cs[ntimes * batch_limit :]  # type: ignore
                ]
        tss = []
        logger.info(f"submit task on {device} for {len(cs)} circuits")  # type: ignore
        time0 = time.time()
        for i, cssi in enumerate(css):
            tss += submit_task(circuit=cssi, shots=shots, device=device, **kws)
            if i < len(css) - 1:
                time.sleep(1.5)
            # TODO(@refraction-ray) whether the sleep time is enough for tquk?
            # incase duplicae request error protection
        l = [t.results() for t in tss]
        time1 = time.time()
        logger.info(
            f"finished collecting count results of {len(cs)} tasks in {round(time1-time0, 4)} seconds"  # type: ignore
        )
        if single is False:
            return l
        return l[0]  # type: ignore

    return run


def sample_expectation_ps(
    c: Circuit,
    x: Optional[Sequence[int]] = None,
    y: Optional[Sequence[int]] = None,
    z: Optional[Sequence[int]] = None,
    shots: int = 1024,
    device: Optional[Device] = None,
    **kws: Any,
) -> float:
    """
    Deprecated, please use :py:func:`tensorcircuit.cloud.wrapper.batch_expectation_ps`.
    """
    logger.warning("This method is deprecated and not maintained, \
        please use `tensorcircuit.cloud.wrapper.batch_expectation_ps` instead")
    c1 = Circuit.from_qir(c.to_qir())
    if x is None:
        x = []
    if y is None:
        y = []
    if z is None:
        z = []
    for i in x:
        c1.H(i)  # type: ignore
    for i in y:
        c1.rx(i, theta=np.pi / 2)  # type: ignore
    if device is None:
        device = get_device()
    t = submit_task(circuit=c1, device=device, shots=shots)
    raw_counts = t.results(blocked=True)  # type: ignore
    x, y, z = list(x), list(y), list(z)
    return counts.expectation(raw_counts, x + y + z)


def reduce_and_evaluate(
    cs: List[Circuit], shots: int, run: Callable[..., Any]
) -> List[counts.ct]:
    reduced_cs = []
    reduced_dict = {}
    recover_dict = {}
    # merge the same circuit, by json which keeps the delays unlike OpenQASM 2
    for j, c in enumerate(cs):
        key = hash(c.to_json())
        if key not in reduced_dict:
            reduced_dict[key] = [j]
            reduced_cs.append(c)
            recover_dict[key] = len(reduced_cs) - 1
        else:
            reduced_dict[key].append(j)

    reduced_raw_counts = run(reduced_cs, shots)
    raw_counts: List[Dict[str, int]] = [None] * len(cs)  # type: ignore
    for i, c in enumerate(cs):
        key = hash(c.to_json())
        raw_counts[i] = reduced_raw_counts[recover_dict[key]]
    return raw_counts


def _qwc_groups(pss: Sequence[Sequence[int]]) -> Tuple[List[List[int]], Tensor]:
    """
    Partition the Pauli strings into qubit-wise commuting groups by greedy coloring
    in the largest-first order, comparing one string with all strings or group
    bases at a time, so that the memory is linear in the number of strings.

    :param pss: Pauli strings of the same length
    :type pss: Sequence[Sequence[int]]
    :return: the indices of the strings in each group, and the measurement basis
        of each group, 0 on the qubits measured by no string of the group
    :rtype: Tuple[List[List[int]], Tensor]
    """
    a = np.array(pss, dtype=np.int8).reshape([len(pss), -1])
    nz = a != 0
    # number of strings each string does not qubit-wise commute with
    degrees = [
        np.count_nonzero(np.any(nz[i] & nz & (a != a[i]), axis=1))
        for i in range(len(a))
    ]
    groups: List[List[int]] = []
    bases = np.zeros_like(a)
    for i in sorted(range(len(a)), key=lambda i: (-degrees[i], i)):
        b = bases[: len(groups)]
        fits = ~np.any(nz[i] & (b != 0) & (b != a[i]), axis=1)
        if fits.any():
            k = int(np.argmax(fits))
            groups[k].append(i)
            # the non-identity entries agree within a group
            bases[k] = np.maximum(bases[k], a[i])
        else:
            bases[len(groups)] = a[i]
            groups.append([i])
    return groups, bases[: len(groups)]


def _flatten(x: Any) -> Tuple[List[Any], Tuple[int, ...]]:
    # the leaves of a nested list in order, and its shape
    leaves, treedef = backend.tree_flatten(x)
    return leaves, np.shape(backend.tree_unflatten(treedef, list(range(len(leaves)))))


def batch_expectation_ps(
    c: Circuit,
    pss: List[List[int]],
    device: Any = None,
    ws: Optional[List[float]] = None,
    shots: int = 8192,
    with_rem: bool = True,
    compile_func: Optional[Callable[[Circuit], Tuple[Circuit, Dict[str, Any]]]] = None,
    batch_limit: int = 64,
    batch_submit_func: Optional[Callable[..., List[counts.ct]]] = None,
    readout_cals: str = "api",
    params: Optional[Union[Dict[Any, Any], Sequence[Dict[Any, Any]]]] = None,
    info: Optional[Dict[str, Any]] = None,
    variants: Any = None,
) -> Union[Any, List[Any]]:
    """
    Unified interface to compute the Pauli string expectation lists or sums via simulation or on real qpu.
    Error mitigation, circuit compilation and Pauli string grouping are all built-in.
    On a device, the Pauli strings are greedily grouped into qubit-wise commuting
    sets, each measured by one circuit, so the expectations in a group share the
    shots, e.g. all ``Z_iZ_{i+1}`` terms of an Ising Hamiltonian in one circuit and
    all ``X_i`` terms in another.

    A ``SymbolCircuit`` is compiled once and bound by each of ``params``, whose
    circuits are all submitted together, e.g. for the parameter shift gradient. To
    keep the compiled circuit, e.g. its layout, over the iterations of a variational
    algorithm, compile it once beforehand and pass it with its ``info``.

    With ``variants``, e.g. the error mitigation methods of
    :py:mod:`tensorcircuit.results.qem.mitigators`, each measurement circuit, i.e.
    compiled and rotated to the measurement basis, is expanded into its variants,
    all submitted together, and the results of the variants are returned on the
    last axes, not combined. With ``device=None``, the same expansion is
    simulated exactly instead, giving the ideal values of the variants, for which
    the circuit compiled beforehand and its ``info`` are needed, the same as those
    run on the device.

    One line access to unlock the whole power or real quantum hardware on quantum cloud.

    :Example:

    .. code-block:: python

        c = tc.Circuit(2)
        c.h(0)
        c.x(1)
        tc.cloud.wrapper.batch_expectation_ps(c, [[1, 0], [0, 3]], device=None)
        # array([ 0.99999994, -0.99999994], dtype=float32)
        tc.cloud.wrapper.batch_expectation_ps(c, [[1, 0], [0, 3]], device="tencent::9gmon")
        # array([ 1.03093477, -1.11715944])

        # a SymbolCircuit compiled once, then bound and run at each iteration
        p = tc.cloud.apis.get_device(device).list_properties()
        csc, info = tc.compiler.DefaultCompiler(properties=p)(sc)
        tc.cloud.wrapper.batch_expectation_ps(
            csc, pss, device=device, ws=ws, info=info, params=[{theta: 0.1}, {theta: 0.2}]
        )

        # twirled and rescaled by the training circuits for each Pauli string
        from tensorcircuit.results.qem import mitigators
        m = mitigators.Chain(mitigators.CDR(2), mitigators.Twirl(8))
        cc, info = tc.compiler.DefaultCompiler(properties=p)(c)
        noisy = tc.cloud.wrapper.batch_expectation_ps(
            cc, pss, device=device, info=info, variants=m
        )
        ideal = tc.cloud.wrapper.batch_expectation_ps(cc, pss, info=info, variants=m)
        m.combine(noisy, ideal)

    :param c: The target circuit to compute expectation, or a ``SymbolCircuit``
        bound by ``params``
    :type c: Circuit
    :param pss: List of Pauli string list, eg. [[0, 1, 0], [2, 3, 3]] represents [X1, Y0Z1Z2].
    :type pss: List[List[int]]
    :param device: The device str or object for quantum cloud module,
        defaults to None, None is for analytical exact simulation
    :type device: Any, optional
    :param ws: List of float to indicate the final return is the weighted sum of Pauli string expectations,
        e.g. [2., -0.3] represents the final results is 2* ``pss`` [0]-0.3* ``pss`` [1]
        defaults to None, None indicate the list of expectations for ``pss`` are all returned
    :type ws: Optional[List[float]], optional
    :param shots: measurement shots for each measured group of Pauli strings, defaults to 8192
    :type shots: int, optional
    :param with_rem: whether enable readout error mitigation for the result, defaults to True
    :type with_rem: bool, optional
    :param readout_cals: where the readout errors of the measured physical qubits
        are from, "api" for the device properties, raising if any is missing, or
        "system" for two calibration circuits run on each call, defaults to "api"
    :type readout_cals: str, optional
    :param params: the values of the free symbols of a ``SymbolCircuit`` ``c`` as
        ``{symbol: value}``, like :py:meth:`tensorcircuit.SymbolCircuit.to_circuit`,
        or a list of them, giving the results stacked by them, defaults to None
    :type params: Optional[Union[Dict[Any, Any], Sequence[Dict[Any, Any]]]], optional
    :param info: the info of ``c`` compiled beforehand, e.g. by ``compile_func``,
        with its ``logical_physical_mapping`` from qubit ``i`` of the Pauli strings
        to the qubit of ``c``, where ``c`` is not compiled again, defaults to None
    :type info: Optional[Dict[str, Any]], optional
    :param variants: an object with ``expand``, e.g. of
        :py:mod:`tensorcircuit.results.qem.mitigators`, or a function, from a
        measurement circuit to a nested list of the circuits run instead, of the
        same shape for all the measurement circuits, defaults to None
    :type variants: Any, optional
    :return: List of Pauli string expectation or a weighted sum float for Pauli strings, depending on ``ws``,
        stacked by ``params`` if a list, with the axes of the ``variants`` last
    :rtype: Union[Any, List[Any]]
    """
    if params is not None and not isinstance(c, SymbolCircuit):
        raise ValueError("`params` are for a SymbolCircuit")
    if device is None and variants is not None and info is None:
        raise ValueError(
            "`variants` without a device need the compiled circuit and its `info`, "
            "the same as run on the device"
        )
    bindings = list(params) if is_sequence(params) else [params]  # type: ignore
    n = len(pss[0])
    mapping = (info or {}).get("logical_physical_mapping") or {i: i for i in range(n)}

    def bound(circuit: Any, binding: Any) -> Any:
        if isinstance(circuit, SymbolCircuit):
            return circuit.to_circuit(binding)
        return circuit

    def evaluate(results: List[Any]) -> Any:
        if ws is not None:
            return backend.convert_to_tensor(sum([w * r for w, r in zip(ws, results)]))
        return backend.stack(results)

    def stacked(results: List[Any]) -> Any:
        return backend.stack(results) if is_sequence(params) else results[0]

    if device is None and variants is None:
        exact = []
        for b in bindings:
            cb = bound(c, b)
            exact.append(
                evaluate(
                    [
                        backend.real(
                            cb.expectation_ps(  # type: ignore
                                **{
                                    k: [mapping[i] for i in v]
                                    for k, v in ps2xyz(ps).items()
                                }
                            )
                        )
                        for ps in pss
                    ]
                )
            )
        return stacked(exact)
    cs = []
    if isinstance(device, str):
        device = get_device(device)

    if info is None:
        if compile_func is None:
            try:
                properties = device.list_properties()
            except (AttributeError, ValueError):
                logger.warning(
                    "device topology unavailable for %r, falling back to "
                    "unconstrained compile",
                    getattr(device, "name", device),
                )
                compile_func = DefaultCompiler()
            else:
                if "bits" in properties and "links" in properties:
                    try:
                        compile_func = DefaultCompiler(properties=properties)
                    except ValueError:
                        logger.warning(
                            "no qubit with known errors on %r, compiling on the "
                            "topology only",
                            getattr(device, "name", device),
                        )
                if compile_func is None:
                    compile_func = DefaultCompiler({"coupling_map": device.topology()})
        c, info = compile_func(c)  # type: ignore
        mapping = info.get("logical_physical_mapping") or mapping  # type: ignore
    info = dict(info, logical_physical_mapping=mapping)  # type: ignore
    groups, bases = _qwc_groups(pss)
    for b in bindings:
        c1 = bound(c, b)
        for basis in bases:
            c2 = Circuit.from_qir(c1.to_qir(), circuit_params={"nqubits": c1._nqubits})
            for j, i in enumerate(basis):
                if i == 1:
                    c2.H(mapping[j])  # type: ignore
                elif i == 2:
                    # sd then H also measures Y, with one physical pulse fewer when
                    # the float32 angle is not compiled as pi/2, no better on hardware
                    c2.rx(mapping[j], theta=np.pi / 2)  # type: ignore
            if any(i in (1, 2) for i in basis):
                c2, _ = simple_compile(c2)
            for i in range(n):
                c2.measure_instruction(mapping[i])
            cs.append(c2)

    # the variants of each measurement circuit in order
    if variants is None:
        flattened = [_flatten(c2) for c2 in cs]
    else:
        expand = getattr(variants, "expand", variants)
        flattened = [_flatten(expand(c2)) for c2 in cs]
    shape = flattened[0][1]
    cs = [ci for leaves, _ in flattened for ci in leaves]
    nv = len(flattened[0][0])
    exps = [[j for j, i in enumerate(ps) if i != 0] for ps in pss]

    if device is None:
        # simulated on the qubits in use only, e.g. of a circuit compiled on a device
        active = sorted(
            {q for ci in cs for d in ci.to_qir() for q in d["index"]}
            | set(mapping.values())
        )
        index = {q: k for k, q in enumerate(active)}
        compact = [ci.initial_mapping(index, len(active)) for ci in cs]

        def value(j: int, i: int) -> Any:
            return backend.real(
                compact[j].expectation_ps(z=[index[mapping[q]] for q in exps[i]])
            )

    else:
        if batch_submit_func is None:
            # the circuits are already on the physical qubits, not to be remapped
            run = batch_submit_template(
                device,
                batch_limit,
                enable_qos_qubit_mapping=False,
                enable_qos_gate_decomposition=False,
                compiling=False,
            )
        else:
            run = batch_submit_func

        mit: Optional[ReadoutMit] = None
        if with_rem:
            mit = ReadoutMit(run)
            qubits = sorted({mapping[i] for i in range(n)})
            if readout_cals == "api":
                mit.cals_from_api(qubits, device)
            elif readout_cals == "system":
                mit.cals_from_system(qubits, shots=shots)
            else:
                raise ValueError("Unknown readout_cals: %s" % readout_cals)

        all_counts = reduce_and_evaluate(cs, shots, run)  # type: ignore

        def value(j: int, i: int) -> Any:
            if mit is not None:
                return mit.expectation(all_counts[j], exps[i], **info)
            return counts.expectation(all_counts[j], exps[i])

    group = {i: g for g, members in enumerate(groups) for i in members}
    outputs = []
    for k in range(len(bindings)):
        results = []
        for i in range(len(pss)):
            j0 = (k * len(bases) + group[i]) * nv
            vs = [value(j0 + v, i) for v in range(nv)]
            results.append(
                vs[0]
                if variants is None
                else backend.reshape(backend.stack(vs), list(shape))
            )
        outputs.append(evaluate(results))
    return stacked(outputs)
