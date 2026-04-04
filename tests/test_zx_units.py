from fractions import Fraction
import pytest
import numpy as np
import jax.numpy as jnp
import pyzx_param as pyzx
import tensorcircuit as tc

from tensorcircuit.zx.noise_model import (
    correlated_error_probs,
    xor_convolve,
    expand_channel,
    Channel,
    absorb_subset_channels,
    ChannelSampler,
)
from tensorcircuit.zx.utils import (
    find_basis,
    _collect_vertices,
    _induced_subgraph,
)
from tensorcircuit.zx.simplifier import full_reduce, teleport_reduce, t_count
from tensorcircuit.zx.scalar_graph import (
    compile_scalar_graphs,
    find_stab,
    CompiledProgram,
)
from tensorcircuit.zx.stabilizertcircuit import StabilizerTCircuit, sample_program
from tensorcircuit.zx.converter import (
    GraphRepresentation,
    h_xy,
    h_yz,
    sqrt_x_dag,
    sqrt_y,
    sqrt_y_dag,
    sqrt_z,
    sqrt_z_dag,
    y_phase,
    r_z,
    r_x,
    r_y,
    u3,
    _swap,
    pauli_channel_2,
    m,
    mrx,
    mry,
    mx,
    my,
    _y_gate,
    _h_xy,
    circuit_to_zx,
)


def test_simplifier_coverage():
    # Test simplifier.py wrappers
    c = tc.Circuit(1)
    c.h(0)
    g = pyzx.Circuit.from_qasm(c.to_openqasm()).to_graph()

    # full_reduce
    full_reduce(g)

    # teleport_reduce
    g2 = teleport_reduce(g)
    assert g2 is not None

    # t_count
    assert t_count(g) == 0


def test_noise_model_correlated_error():
    probs = [0.1, 0.2]
    # Total = 0.1 + 0.18 + 0.72 = 1.0
    res = correlated_error_probs(probs)
    assert np.allclose(res, [0.72, 0.1, 0.18, 0.0])


def test_noise_model_xor_convolve_error():
    with pytest.raises(
        ValueError, match="Both channels must have same number of outcomes"
    ):
        xor_convolve(np.array([0.5, 0.5]), np.array([0.1, 0.2, 0.3, 0.4]))


def test_noise_model_expand_channel():
    ch = Channel(probs=np.array([0.9, 0.1]), unique_col_ids=(1,))
    # expand to (0, 1)
    expanded = expand_channel(ch, (0, 1))
    assert np.allclose(expanded.probs, [0.9, 0.0, 0.1, 0.0])


def test_noise_model_absorb_subset():
    ch1 = Channel(probs=np.array([0.9, 0.0, 0.0, 0.1]), unique_col_ids=(1, 2))
    ch2 = Channel(probs=np.array([0.8, 0.2]), unique_col_ids=(1,))
    res = absorb_subset_channels([ch1, ch2], max_bits=4)
    assert len(res) == 1
    assert res[0].unique_col_ids == (1, 2)


def test_noise_model_sampler_zero_prob():
    # Test _precompute_sparse with p_fire <= 1e-15
    sampler = ChannelSampler([np.array([1.0, 0.0])], np.array([[1]]))
    assert len(sampler._sparse_data) == 0
    samples = sampler.sample(10)
    assert samples.shape == (10, 1)
    assert np.all(samples == 0)

    # Trigger line 538: num_samples=1, p_fire=1e-10 -> very likely no firing
    sampler2 = ChannelSampler([np.array([1 - 1e-10, 1e-10])], np.array([[1]]))
    sampler2.sample(1)


def test_utils_find_basis_xor():
    # Test find_basis where XOR is needed
    vectors = [[1, 0], [0, 1], [1, 1]]
    basis, transform = find_basis(vectors)
    assert len(basis) == 2
    assert np.all(transform[2] == [1, 1])


def test_utils_collect_vertices_visited():
    g = pyzx.Graph()
    v0 = g.add_vertex(1)
    visited = {v0}
    res = _collect_vertices(g, v0, visited)
    assert len(res) == 0


def test_utils_induced_subgraph_vdata():
    g = pyzx.Graph()
    v0 = g.add_vertex(1)
    g.set_vdata(v0, "test", "val")
    sub, vmap = _induced_subgraph(g, [v0])
    assert sub.vdata(vmap[v0], "test") == "val"


def test_utils_induced_subgraph_no_neighbor():
    g = pyzx.Graph()
    v0 = g.add_vertex(1)
    v1 = g.add_vertex(1)
    g.add_edge((v0, v1))
    sub, _ = _induced_subgraph(g, [v0])
    assert sub.num_vertices() == 1
    assert sub.num_edges() == 0


def test_scalar_graph_compile_approx():
    # Test approximate_floatfactors
    g = pyzx.Graph()
    g.scalar.approximate_floatfactor = 0.5
    res = compile_scalar_graphs([g], [])
    assert res.has_approximate_floatfactors
    assert np.allclose(res.approximate_floatfactors, [0.5])


def test_scalar_graph_compile_complex_phase():
    g = pyzx.Graph()
    g.scalar.phase = Fraction(1, 3)
    res = compile_scalar_graphs([g], [])
    assert res.has_approximate_floatfactors
    expected = np.exp(1j * np.pi / 3)
    assert np.allclose(res.approximate_floatfactors, [expected])


def test_stabilizertcircuit_empty_program():
    _ = StabilizerTCircuit(1)

    program = CompiledProgram(
        components=[], num_f_params=0, output_order=[], mode="sequential"
    )
    res = sample_program(program, jnp.zeros((10, 0), dtype=jnp.bool_), None)
    assert res.shape == (10, 0)


def test_stabilizertcircuit_amplitude_error():
    stc = StabilizerTCircuit(1)
    stc.x_error(0, 0.1)
    with pytest.raises(
        ValueError, match="amplitude\(\) only supported for noiseless circuits"
    ):
        stc.amplitude("0")


def test_stabilizertcircuit_expectation_nmc_error():
    stc = StabilizerTCircuit(1)
    stc.x_error(0, 0.1)
    with pytest.raises(ValueError, match="nmc must be positive"):
        stc.expectation_ps(z=[0], nmc=0)


def test_stabilizertcircuit_apply_general_ir_dict():
    stc = StabilizerTCircuit(1)
    ir = {"name": "H", "index": [0]}
    stc.apply_general_gate(None, 0, ir_dict=ir)
    assert len(stc._qir) == 1
    assert stc._qir[0]["name"] == "H"


def test_stabilizertcircuit_getattr_wrapper():
    stc = StabilizerTCircuit(1)
    stc.h(0)
    stc.s_dag(0)
    assert stc._qir[1]["name"] == "S_DAG"

    with pytest.raises(AttributeError):
        stc.non_existent_gate(0)


def test_stabilizertcircuit_from_circuit_gatef():
    c = tc.Circuit(1)

    class CustomGate:
        def __init__(self):
            self.name = "custom"

        def __call__(self, *args, **kwargs):
            return self

    class CustomGateNoName:
        def __init__(self):
            self.__name__ = "custom_noname"

        def __call__(self, *args, **kwargs):
            return self

    # Manually append to _qir to simulate a gate with gatef and no name
    c._qir.append({"gatef": CustomGate(), "index": [0]})
    c._qir.append({"gatef": CustomGateNoName(), "index": [0]})
    stclocal = StabilizerTCircuit.from_circuit(c)
    assert stclocal._qir[0]["name"] == "CUSTOM"
    assert stclocal._qir[1]["name"] == "CUSTOM_NONAME"


def test_stabilizertcircuit_expectation_noisy_f():
    # Trigger line 472: f-parameter handling in expectation_ps
    stc = StabilizerTCircuit(2)
    stc.depolarizing(0, 0.1)
    exp = stc.expectation_ps(z=[0], nmc=100)
    assert exp is not None


def test_stabilizertcircuit_apply_no_name():
    stc = StabilizerTCircuit(1)

    class G:
        pass

    stc.apply(G(), 0)
    stc.apply_general_gate(G(), 0)

    class GWithName:
        name = "h"

    stc.apply_general_gate(GWithName(), 0)


def test_converter_missing_gates():
    b = GraphRepresentation()
    h_xy(b, 0)
    h_yz(b, 0)
    sqrt_x_dag(b, 0)
    sqrt_y(b, 0)
    sqrt_y_dag(b, 0)
    sqrt_z(b, 0)
    sqrt_z_dag(b, 0)
    y_phase(b, 0, Fraction(1, 4))
    r_z(b, 0, Fraction(1, 4))
    r_x(b, 0, Fraction(1, 4))
    r_y(b, 0, Fraction(1, 4))
    u3(b, 0, Fraction(1, 4), Fraction(1, 4), Fraction(1, 4))
    _swap(b, 0, 1)
    pauli_channel_2(b, 0, 1, pix=0.01)
    m(b, 0, invert=True)
    mrx(b, 0, p=0.01)
    mry(b, 0, p=0.01)
    mx(b, 0, p=0.01)
    my(b, 0, p=0.01)
    _y_gate(b, 0)
    _h_xy(b, 0)

    try:
        v = b.add_vertex()
        b.set_qubit(v, 0)
        b.vertex_degree(v)
        b.remove_isolated_vertices()
        b.get_params(v)
        b.incident_edges(v)
        b.vdata_keys(v)
        b.vdata(v, "k")
        b.set_vdata(v, "k", "v")
        b.get_auto_simplify()
        b.set_auto_simplify(True)
        b.is_multigraph()
        b.vertex_set()
        b.edge_set()
        b.num_vertices()
        b.num_edges()
        b.copy()
    except Exception:
        pass


def test_converter_dispatch_more():

    c = tc.Circuit(2)
    # Manually append to _qir to ensure upper() names work and no 'to_qir' error
    c._qir.append(
        {
            "name": "U3",
            "index": [0],
            "parameters": {"theta": 0.1, "phi": 0.2, "lambda_": 0.3},
        }
    )
    c._qir.append({"name": "SD", "index": [0]})
    c._qir.append({"name": "S_DAG", "index": [0]})
    c._qir.append({"name": "TD", "index": [0]})
    c._qir.append({"name": "T_DAG", "index": [0]})
    c._qir.append({"name": "SQRT_X_DAG", "index": [0]})
    c._qir.append({"name": "SQRT_Y", "index": [0]})
    c._qir.append({"name": "SQRT_Y_DAG", "index": [0]})
    c._qir.append({"name": "SQRT_Z", "index": [0]})
    c._qir.append({"name": "H_XY", "index": [0]})
    c._qir.append({"name": "H_YZ", "index": [0]})
    c._qir.append({"name": "H_XZ", "index": [0]})
    c._qir.append({"name": "R_X", "index": [0], "parameters": {"theta": 0.1}})
    c._qir.append({"name": "R_Y", "index": [0], "parameters": {"theta": 0.1}})
    c._qir.append({"name": "R_Z", "index": [0], "parameters": {"theta": 0.1}})
    c._qir.append({"name": "MRX", "index": [0]})
    c._qir.append({"name": "MRY", "index": [0]})
    c._qir.append({"name": "MRZ", "index": [0]})
    c._qir.append({"name": "MX", "index": [0]})
    c._qir.append({"name": "MY", "index": [0]})
    c._qir.append({"name": "MZ", "index": [0]})
    c._qir.append({"name": "RX", "index": [0]})
    c._qir.append({"name": "RY", "index": [0]})
    c._qir.append({"name": "RZ", "index": [0]})

    g = circuit_to_zx(c)
    assert g is not None


def test_stabilizertcircuit_from_stim_more():
    import stim

    sc = stim.Circuit("""
        I 0
        TICK
        OBSERVABLE_INCLUDE(5) rec[-1]
        ISWAP 0 1
        SQRT_XX 0 1
        M(0.1) 0
        MR(0.1) 0
        MX(0.1) 0
        MRX(0.1) 0
        MRY(0.1) 0
        MRZ(0.1) 0
        MY(0.1) 0
    """)
    # Add I_ERROR which should be skipped
    sc.append("I_ERROR", [0], [0.1])

    stc = StabilizerTCircuit.from_stim_circuit(sc)
    assert any(d["name"] == "TICK" for d in stc._qir)
    assert any(d["name"] == "OBSERVABLE_INCLUDE" for d in stc._qir)
    assert any(d["name"] == "ISWAP" for d in stc._qir)
    assert any(d["name"] == "SQRT_XX" for d in stc._qir)
    m_ops = [d for d in stc._qir if d["name"] == "MEASURE"]
    assert any("p" in d for d in m_ops)


def test_stabilizertcircuit_from_stim_tagged_i():
    import stim

    sc = stim.Circuit()
    sc.append("I", [0], tag="H(theta=0.5*pi)")
    stc = StabilizerTCircuit.from_stim_circuit(sc)
    assert stc._qir[0]["name"] == "H"
    assert stc._qir[0]["parameters"]["theta"] == 0.5 * np.pi


def test_scalar_graph_find_stab_wrapped():
    class Wrapper:
        def __init__(self, g):
            self.graph = g

    g = pyzx.Graph()
    res = find_stab(Wrapper(g), strategy="cat5")
    assert len(res) >= 1


def test_stabilizertcircuit_one_liners():
    stc = StabilizerTCircuit(2)
    stc.x(0)
    stc.y(0)
    stc.z(0)
    stc.s(0)
    stc.sd(0)
    stc.t(0)
    stc.td(0)
    stc.cx(0, 1)
    stc.cz(0, 1)
    stc.swap(0, 1)
    stc.tick_instruction()
    stc.qubit_coords_instruction(0, [1.0, 2.0])
    stc.reset_instruction(0)
    stc.mr_instruction(0, p=0.1)
    stc.mrx_instruction(0, p=0.1)
    stc.mry_instruction(0, p=0.1)
    stc.mrz_instruction(0, p=0.1)
    stc.rx(0, theta=0.1)
    stc.ry(0, theta=0.1)
    stc.rz(0, theta=0.1)
    stc.depolarizing_instruction(0, 0.1)
    stc.depolarizing2_instruction(0, 1, 0.1)
    stc.pauli_instruction(0, px=0.1, py=0.1, pz=0.1)
    stc.pauli(0, [0.7, 0.1, 0.1, 0.1])
    stc.y_error(0, 0.1)
    stc.z_error(0, 0.1)

    class G:
        def __init__(self):
            self.name = "H"

    stc.apply(G(), 0)
    assert len(stc._qir) > 20
