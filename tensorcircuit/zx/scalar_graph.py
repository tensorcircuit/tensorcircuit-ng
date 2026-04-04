"""
Decomposition of ZX graphs into scalar graphs and JAX-compatible IR.
"""

from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple, cast

import jax.numpy as jnp
import numpy as np
from jax import Array
import pyzx_param as zx
from pyzx_param.graph.scalar import DyadicNumber
from pyzx_param.simulate import DecompositionStrategy

from ..cons import dtypestr, idtypestr
from .utils import get_params, connected_components


class CompiledScalarGraphs(NamedTuple):
    num_graphs: int
    n_params: int
    a_const_phases: Array
    a_param_bits: Array
    a_num_terms: Array
    b_term_types: Array
    b_param_bits: Array
    c_const_bits_a: Array
    c_param_bits_a: Array
    c_const_bits_b: Array
    c_param_bits_b: Array
    d_const_alpha: Array
    d_const_beta: Array
    d_param_bits_a: Array
    d_param_bits_b: Array
    d_num_terms: Array
    phase_indices: Array
    has_approximate_floatfactors: bool
    approximate_floatfactors: Array
    power2: Array
    floatfactor: Array


def compile_scalar_graphs(g_list: list[Any], params: list[str]) -> CompiledScalarGraphs:
    """
    Compile a list of ZX scalar graphs into a JAX-compatible IR format.

    :param g_list: List of ZX scalar graphs.
    :type g_list: list[Any]
    :param params: List of parameter names (e.g., 'f0', 'm0').
    :type params: list[str]
    :return: Compiled scalar graphs in IR format.
    :rtype: CompiledScalarGraphs
    """
    # Filter out zero graphs but keep at least one to avoid num_graphs=0
    active_graphs = [g for g in g_list if not g.scalar.is_zero]
    if not active_graphs and len(g_list) > 0:
        # Keep the first graph even if it's zero
        g_list = [g_list[0]]
    else:
        g_list = active_graphs

    n_params, num_graphs = len(params), len(g_list)
    char_to_idx = {char: i for i, char in enumerate(params)}

    # Type A
    a_terms: List[List[Tuple[int, List[int]]]] = [[] for _ in range(num_graphs)]
    for i, g in enumerate(g_list):
        for term in range(len(g.scalar.phasenodevars)):
            bitstr = [0] * n_params
            for v in g.scalar.phasenodevars[term]:
                if v in char_to_idx:
                    bitstr[char_to_idx[v]] = 1

            a_terms[i].append((int(g.scalar.phasenodes[term] * 4), bitstr))
    a_num_terms = np.array([len(t) for t in a_terms], dtype=idtypestr)
    max_a = int(a_num_terms.max()) if a_num_terms.size else 0
    a_const_phases = np.zeros((num_graphs, max_a), dtype=np.uint8)
    a_param_bits = np.zeros((num_graphs, max_a, n_params), dtype=np.uint8)
    for i, a_terms_list in enumerate(a_terms):
        for j, (p, b) in enumerate(a_terms_list):
            a_const_phases[i, j], a_param_bits[i, j] = p, b

    # Type B
    b_terms: List[List[Tuple[int, List[int]]]] = [[] for _ in range(num_graphs)]
    for i, g in enumerate(g_list):
        bitstr_to_j: Dict[Tuple[int, ...], int] = defaultdict(int)
        for j in [1, 3]:
            if j not in g.scalar.phasevars_halfpi:
                continue
            for term in g.scalar.phasevars_halfpi[j]:
                bitstr = [0] * n_params
                for v in cast(Sequence[str], term):
                    if v in char_to_idx:
                        bitstr[char_to_idx[v]] = 1
                bitstr_to_j[tuple(bitstr)] = (bitstr_to_j[tuple(bitstr)] + j) % 4
        for b_key, val in bitstr_to_j.items():
            if val != 0:
                b_terms[i].append((val * 2, list(b_key)))
    max_b = max((len(t) for t in b_terms), default=0)
    b_term_types = np.zeros((num_graphs, max_b), dtype=np.uint8)
    b_param_bits = np.zeros((num_graphs, max_b, n_params), dtype=np.uint8)
    for i, b_terms_list in enumerate(b_terms):
        for j, (p, b) in enumerate(b_terms_list):
            b_term_types[i, j], b_param_bits[i, j] = p, b

    # Type C
    c_terms: List[List[Tuple[Any, ...]]] = [[] for _ in range(num_graphs)]
    for i, g in enumerate(g_list):
        for p_set in g.scalar.phasevars_pi_pair:
            c_bits = []
            for ps in p_set:
                c_bits.append(1 if "1" in ps else 0)
                p_bits = [0] * n_params
                for p in ps:
                    p_str = str(p)
                    if p_str != "1" and p_str in char_to_idx:
                        p_bits[char_to_idx[p_str]] = 1
                c_bits.append(cast(int, p_bits))  # type: ignore
            c_terms[i].append(tuple(c_bits))
    max_c = max((len(t) for t in c_terms), default=0)
    c_const_bits_a = np.zeros((num_graphs, max_c), dtype=np.uint8)
    c_param_bits_a = np.zeros((num_graphs, max_c, n_params), dtype=np.uint8)
    c_const_bits_b = np.zeros((num_graphs, max_c), dtype=np.uint8)
    c_param_bits_b = np.zeros((num_graphs, max_c, n_params), dtype=np.uint8)
    for i, c_terms_list in enumerate(c_terms):
        for j, (ca, pa, cb, pb) in enumerate(c_terms_list):
            (
                c_const_bits_a[i, j],
                c_param_bits_a[i, j],
                c_const_bits_b[i, j],
                c_param_bits_b[i, j],
            ) = (ca, pa, cb, pb)

    # Type D
    d_terms: List[List[Tuple[int, int, List[int], List[int]]]] = [
        [] for _ in range(num_graphs)
    ]
    for i, g in enumerate(g_list):
        for pp in g.scalar.phasepairs:
            pa, pb = [0] * n_params, [0] * n_params
            for v in pp.paramsA:
                if v in char_to_idx:
                    pa[char_to_idx[v]] = 1
            for v in pp.paramsB:
                if v in char_to_idx:
                    pb[char_to_idx[v]] = 1
            d_terms[i].append((int(pp.alpha), int(pp.beta), pa, pb))
    d_num_terms = np.array([len(t) for t in d_terms], dtype=idtypestr)
    max_d = int(d_num_terms.max()) if d_num_terms.size else 0
    d_const_alpha = np.zeros((num_graphs, max_d), dtype=np.uint8)
    d_const_beta = np.zeros((num_graphs, max_d), dtype=np.uint8)
    d_param_bits_a = np.zeros((num_graphs, max_d, n_params), dtype=np.uint8)
    d_param_bits_b = np.zeros((num_graphs, max_d, n_params), dtype=np.uint8)
    for i, d_terms_list in enumerate(d_terms):
        for j, (ca, cb, pa, pb) in enumerate(d_terms_list):
            (
                d_const_alpha[i, j],
                d_const_beta[i, j],
                d_param_bits_a[i, j],
                d_param_bits_b[i, j],
            ) = (ca, cb, pa, pb)

    # Static Data
    for g in g_list:
        if not isinstance(
            g.scalar.phase, Fraction
        ) or g.scalar.phase.denominator not in [
            1,
            2,
            4,
        ]:
            g.scalar.approximate_floatfactor *= np.exp(
                1j * float(g.scalar.phase) * np.pi
            )
            g.scalar.phase = Fraction(0, 1)

    exact_floatfactor, power2 = [], []
    for g in g_list:
        if g.scalar.is_zero:
            power2.append(0)
            exact_floatfactor.append([0, 0, 0, 0])
            continue
        dn, p_sqrt2 = g.scalar.floatfactor.copy(), g.scalar.power2
        if p_sqrt2 % 2 != 0:
            p_sqrt2 -= 1
            dn *= DyadicNumber(k=0, a=0, b=1, c=0, d=1)
        p_sqrt2 -= 2 * dn.k
        dn.k = 0
        power2.append(p_sqrt2 // 2)
        exact_floatfactor.append([dn.a, dn.b, dn.c, dn.d])

    return CompiledScalarGraphs(
        num_graphs=num_graphs,
        n_params=n_params,
        a_const_phases=jnp.array(a_const_phases),
        a_param_bits=jnp.array(a_param_bits),
        a_num_terms=jnp.array(a_num_terms),
        b_term_types=jnp.array(b_term_types),
        b_param_bits=jnp.array(b_param_bits),
        c_const_bits_a=jnp.array(c_const_bits_a),
        c_param_bits_a=jnp.array(c_param_bits_a),
        c_const_bits_b=jnp.array(c_const_bits_b),
        c_param_bits_b=jnp.array(c_param_bits_b),
        d_const_alpha=jnp.array(d_const_alpha),
        d_const_beta=jnp.array(d_const_beta),
        d_param_bits_a=jnp.array(d_param_bits_a),
        d_param_bits_b=jnp.array(d_param_bits_b),
        d_num_terms=jnp.array(d_num_terms),
        phase_indices=jnp.array(
            [int(float(g.scalar.phase) * 4) for g in g_list], dtype=jnp.uint8
        ),
        has_approximate_floatfactors=any(
            g.scalar.approximate_floatfactor != 1.0 for g in g_list
        ),
        approximate_floatfactors=jnp.array(
            [g.scalar.approximate_floatfactor for g in g_list], dtype=dtypestr
        ),
        power2=jnp.array(power2, dtype=idtypestr),
        floatfactor=jnp.array(exact_floatfactor, dtype=idtypestr),
    )


class CompiledComponent(NamedTuple):
    output_indices: list[int]
    f_selection: list[int]
    compiled_scalar_graphs: list[CompiledScalarGraphs]


class CompiledProgram(NamedTuple):
    components: list[CompiledComponent]
    num_f_params: int
    output_order: list[int]
    mode: str


def _decompose(graphs: List[Any], count_fn: Any, replace_fn: Any) -> List[Any]:
    results: list[Any] = []
    for graph in graphs:
        if count_fn(graph) == 0:
            results.append(graph)
            continue

        gsum = replace_fn(graph.copy())
        for g in gsum.graphs:
            zx.full_reduce(g, paramSafe=True)
            if g.scalar.is_zero:
                if len(results) > 0:
                    continue
            results.extend(_decompose([g], count_fn, replace_fn))
    return results


def find_stab_magic(graphs: List[Any], strategy: DecompositionStrategy) -> list[Any]:
    return _decompose(
        list(graphs),
        count_fn=zx.simplify.tcount,
        replace_fn=lambda g: zx.simulate.replace_magic_states(
            g, pick_random=False, strategy=strategy
        ),
    )


def find_stab_u3(graphs: List[Any], strategy: DecompositionStrategy) -> list[Any]:
    return _decompose(
        list(graphs),
        count_fn=zx.simplify.u3_count,
        replace_fn=lambda g: zx.simulate.replace_u3_states(g, strategy=strategy),
    )


def find_stab(graph: Any, strategy: DecompositionStrategy) -> List[Any]:
    """
    Decompose a ZX graph into a sum of stabilizer graphs.

    :param graph: The ZX graph to decompose.
    :type graph: Any
    :param strategy: Decomposition strategy for T gates.
    :type strategy: DecompositionStrategy
    :return: List of stabilizer graphs.
    :rtype: List[Any]
    """
    if hasattr(graph, "graph") and not hasattr(graph, "add_vertex"):
        graph = graph.graph
    zx.full_reduce(graph, paramSafe=True)
    graphs = find_stab_u3([graph], strategy=strategy)
    return find_stab_magic(graphs, strategy=strategy)


def _get_f_indices(graph: Any) -> list[int]:
    all_params = get_params(graph)
    return sorted([int(p[1:]) for p in all_params if p.startswith("f")])


def _remove_phase_terms(graph: Any) -> None:
    graph.scalar.phasevars_halfpi = dict()
    graph.scalar.phasevars_pi_pair = []


def _compile_component(
    component: Any,
    f_indices_global: list[int],
    mode: str,
    strategy: DecompositionStrategy = "cat5",
) -> CompiledComponent:
    graph = component.graph
    output_indices = component.output_indices
    num_component_outputs = len(graph.outputs())

    # f_selection: subset of global f-indices used by this component
    component_f_set = set(_get_f_indices(graph))
    f_selection = [i for i in f_indices_global if i in component_f_set]

    if mode == "sequential":
        outputs_to_plug = list(range(num_component_outputs + 1))
    else:  # joint
        outputs_to_plug = [0, num_component_outputs]

    # Plug outputs and compile each graph
    compiled_graphs: list[CompiledScalarGraphs] = []

    component_m_chars = [f"m{i}" for i in output_indices]
    plugged_graphs = _plug_outputs(graph, component_m_chars, outputs_to_plug)

    # Track power2 for balancing scalar magnitudes
    power2_base: int | None = None

    for num_m_plugged, plugged_graph in zip(outputs_to_plug, plugged_graphs):
        g_copy = plugged_graph.copy()
        if hasattr(plugged_graph, "track_phases"):
            g_copy.track_phases = plugged_graph.track_phases
        if hasattr(plugged_graph, "merge_vdata"):
            g_copy.merge_vdata = plugged_graph.merge_vdata
        zx.full_reduce(g_copy, paramSafe=True)
        g_copy.normalize()

        # Balance power2 across graphs to avoid overflow/underflow
        if power2_base is None:
            power2_base = g_copy.scalar.power2
        g_copy.scalar.add_power(-power2_base)

        # Remove parametrized global phase terms. Global phases only matter once we
        # have started stabilizer rank decomposition.
        _remove_phase_terms(g_copy)

        # Parameter names: all f-params + m-params plugged so far
        param_names = [f"f{i}" for i in f_selection]
        param_names += [f"m{output_indices[j]}" for j in range(num_m_plugged)]

        # Perform stabilizer rank decomposition and compile
        g_list = find_stab(g_copy, strategy=strategy)

        if len(g_list) == 1:
            # This is a Clifford graph, we can clear the global phase terms
            _remove_phase_terms(g_list[0])

        compiled = compile_scalar_graphs(g_list, param_names)
        compiled_graphs.append(compiled)

    return CompiledComponent(
        output_indices=output_indices,
        f_selection=f_selection,
        compiled_scalar_graphs=compiled_graphs,
    )


def _plug_outputs(
    graph: Any,
    m_chars: list[str],
    outputs_to_plug: list[int],
) -> list[Any]:
    """Create graphs with specified numbers of outputs plugged.

    Args:
        graph: The component graph.
        m_chars: The m-parameter names for this component's outputs.
        outputs_to_plug: List of integers specifying how many outputs to plug
            for each graph. E.g., [0, 1, 2, 3] creates 4 graphs.

    Returns:
        List of graphs with outputs plugged according to outputs_to_plug.

    """
    graphs: list[Any] = []
    num_outputs = len(graph.outputs())

    for num_plugged in outputs_to_plug:
        g = graph.copy()
        if hasattr(graph, "track_phases"):
            g.track_phases = graph.track_phases
        if hasattr(graph, "merge_vdata"):
            g.merge_vdata = graph.merge_vdata
        output_vertices = list(g.outputs())

        # Plug outputs either with an X vertex with phase m_char[i]
        # or with a Z vertex. Z vertices are equal to |0> + |1> and therefore
        # implement a trace.
        effect = "0" * num_plugged + "+" * (num_outputs - num_plugged)
        g.apply_effect(effect)
        for i, v in enumerate(output_vertices[:num_plugged]):
            g.set_phase(v, m_chars[i])

        # Compensate power for trace of unplugged outputs
        g.scalar.add_power(num_outputs - num_plugged)

        graphs.append(g)

    return graphs


def compile_program(
    prepared: Any,
    mode: str,
    strategy: DecompositionStrategy = "cat5",
) -> CompiledProgram:
    """
    Compile a prepared sampling graph into a multicomponent program.

    :param prepared: The prepared sampling graph metadata.
    :type prepared: Any
    :param mode: Compilation mode, currently only 'sequential' is supported.
    :type mode: str
    :param strategy: Decomposition strategy for T gates, defaults to "cat5".
    :type strategy: DecompositionStrategy, optional
    :return: The compiled program metadata.
    :rtype: CompiledProgram
    """
    components = connected_components(prepared.graph)

    f_indices_global = _get_f_indices(prepared.graph)

    compiled_components: list[CompiledComponent] = []
    output_order: list[int] = []

    sorted_components = sorted(components, key=lambda c: len(c.output_indices))

    for component in sorted_components:
        compiled = _compile_component(
            component=component,
            f_indices_global=f_indices_global,
            mode=mode,
            strategy=strategy,
        )
        compiled_components.append(compiled)
        output_order.extend(component.output_indices)

    return CompiledProgram(
        components=compiled_components,
        num_f_params=len(f_indices_global),
        output_order=output_order,
        mode=mode,
    )
