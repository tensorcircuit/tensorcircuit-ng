"""
Quantum circuit: common methods for all circuit classes as MixIn

Note:
  - Supports qubit (d = 2) and qudit (d >= 2) systems.
  - For string-encoded samples/counts when d <= 36, digits use base-d characters 0-9A-Z (A = 10, ..., Z = 35).
"""

# pylint: disable=invalid-name

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from functools import partial
import logging

import numpy as np
import graphviz
import tensornetwork as tn

from . import gates
from .quantum import (
    QuOperator,
    QuVector,
    correlation_from_samples,
    correlation_from_counts,
    measurement_counts,
    sample_int2bin,
    sample2all,
    _infer_num_sites,
    _decode_basis_label,
    onehot_d_tensor,
)
from .abstractcircuit import AbstractCircuit
from .cons import npdtype, backend, dtypestr, contractor, rdtypestr
from .simplify import _split_two_qubit_gate
from .utils import arg_alias

logger = logging.getLogger(__name__)

Gate = gates.Gate
Tensor = Any


class BaseCircuit(AbstractCircuit):
    _nodes: List[tn.Node]
    _front: List[tn.Edge]
    is_dm: bool
    split: Optional[Dict[str, Any]]

    is_mps = False

    @staticmethod
    def all_zero_nodes(n: int, prefix: str = "qb-", dim: int = 2) -> List[tn.Node]:
        prefix = "qd-" if dim > 2 else prefix
        l = [0.0] * dim
        l[0] = 1.0
        nodes = [
            tn.Node(
                np.array(
                    l,
                    dtype=npdtype,
                ),
                name=prefix + str(x),
            )
            for x in range(n)
        ]
        return nodes

    @staticmethod
    def front_from_nodes(nodes: List[tn.Node]) -> List[tn.Edge]:
        return [n.get_edge(0) for n in nodes]

    def _tensors_to_nodes(
        self, tensors: Sequence[Tensor]
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        """
        Internal method to convert a sequence of MPS tensors to a list of nodes and front edges.
        (bond-left, physical, bond-right) order is assumed for MPS tensors.

        :param tensors: A sequence of tensors representing an MPS.
        :type tensors: Sequence[Tensor]
        :return: A tuple of (all nodes including dummy boundary nodes, front/physical edges).
        :rtype: Tuple[List[tn.Node], List[tn.Edge]]
        """
        nodes = [
            tn.Node(backend.cast(backend.convert_to_tensor(t), dtypestr))
            for t in tensors
        ]
        for i in range(len(nodes) - 1):
            nodes[i].get_edge(2) ^ nodes[i + 1].get_edge(0)

        q_nodes = nodes
        all_nodes = list(q_nodes)
        for i, axis in zip([0, -1], [0, 2]):
            if q_nodes[i].get_edge(axis).dimension == 1:
                dummy = tn.Node(
                    backend.cast(backend.convert_to_tensor([1.0]), dtypestr)
                )
                q_nodes[i].get_edge(axis) ^ dummy.get_edge(0)
                all_nodes.append(dummy)

        front = [q_nodes[i].get_edge(1) for i in range(len(tensors))]
        return all_nodes, front

    @staticmethod
    def coloring_nodes(
        nodes: Sequence[tn.Node], is_dagger: bool = False, flag: str = "inputs"
    ) -> None:
        r"""
        Tag nodes with metadata used for casual lightcone simplification and tracing.

        :param nodes: A sequence of tensornetwork nodes to tag.
        :type nodes: Sequence[tn.Node]
        :param is_dagger: Whether the nodes represent conjugate operations (U^\dagger),
            defaults to False.
        :type is_dagger: bool, optional
        :param flag: A label for the node type (e.g., "gate", "inputs", "operator"),
            defaults to "inputs".
        :type flag: str, optional
        """
        for node in nodes:
            node.is_dagger = is_dagger
            node.flag = flag
            node.id = id(node)

    @staticmethod
    def coloring_copied_nodes(
        nodes: Sequence[tn.Node],
        nodes0: Sequence[tn.Node],
        is_dagger: bool = True,
        flag: str = "inputs",
    ) -> None:
        """
        Tag copied nodes while preserving the original node's identity for lightcone cancellation.

        :param nodes: A sequence of newly copied nodes.
        :type nodes: Sequence[tn.Node]
        :param nodes0: The sequence of original nodes from which `nodes` were copied.
        :type nodes0: Sequence[tn.Node]
        :param is_dagger: Whether the copied nodes represent conjugate operations,
            defaults to True.
        :type is_dagger: bool, optional
        :param flag: A label for the node type, defaults to "inputs".
        :type flag: str, optional
        """
        for node, n0 in zip(nodes, nodes0):
            node.is_dagger = is_dagger
            node.flag = flag
            node.id = getattr(n0, "id", id(n0))

    @staticmethod
    def copy_nodes(
        nodes: Sequence[tn.Node],
        dangling: Optional[Sequence[tn.Edge]] = None,
        conj: Optional[bool] = False,
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        """
        copy all nodes and dangling edges correspondingly

        :return:
        """
        ndict, edict = tn.copy(nodes, conjugate=conj)
        newnodes = []
        for n in nodes:
            newn = ndict[n]
            newn.is_dagger = conj
            newn.flag = getattr(n, "flag", "") + "copy"
            newn.id = getattr(n, "id", id(n))
            newnodes.append(newn)
        newfront = []
        if not dangling:
            dangling = []
            for n in nodes:
                dangling.extend([e for e in n])
        for e in dangling:
            newfront.append(edict[e])
        return newnodes, newfront

    def _copy(
        self, conj: Optional[bool] = False
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        return self.copy_nodes(self._nodes, self._front, conj)

    def apply_general_gate(
        self,
        gate: Union[Gate, QuOperator],
        *index: int,
        name: Optional[str] = None,
        split: Optional[Dict[str, Any]] = None,
        mpo: bool = False,
        diagonal: bool = False,
        ir_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        if name is None:
            name = ""
        gate_dict = {
            "gate": gate,
            "index": index,
            "name": name,
            "split": split,
            "mpo": mpo,
            "diagonal": diagonal,
        }
        if ir_dict is not None:
            ir_dict.update(gate_dict)
        else:
            ir_dict = gate_dict
        self._qir.append(ir_dict)
        assert len(index) == len(set(index))
        index = tuple(i if i >= 0 else self._nqubits + i for i in index)
        noe = len(index)
        nq = self._nqubits
        applied = False
        split_conf = None
        if split is not None:
            split_conf = split
        elif self.split is not None:
            split_conf = self.split

        if not (mpo or diagonal):
            assert isinstance(gate, tn.Node)
            if (split_conf is not None) and noe == 2:
                results = _split_two_qubit_gate(gate, **split_conf)
                # max_err cannot be jax jitted
                if results is not None:
                    n1, n2, is_swap = results
                    self.coloring_nodes([n1, n2], flag="gate")
                    # n1.flag = "gate"
                    # n1.is_dagger = False
                    n1.name = name
                    # n1.id = id(n1)
                    # n2.flag = "gate"
                    # n2.is_dagger = False
                    # n2.id = id(n2)
                    n2.name = name
                    if is_swap is False:
                        n1[1] ^ self._front[index[0]]
                        n2[2] ^ self._front[index[1]]
                        self._nodes.append(n1)
                        self._nodes.append(n2)
                        self._front[index[0]] = n1[0]
                        self._front[index[1]] = n2[1]
                        if self.is_dm:
                            [n1l, n2l], _ = self.copy_nodes([n1, n2], conj=True)
                            n1l[1] ^ self._front[index[0] + nq]
                            n2l[2] ^ self._front[index[1] + nq]
                            self._nodes.append(n1l)
                            self._nodes.append(n2l)
                            self._front[index[0] + nq] = n1l[0]
                            self._front[index[1] + nq] = n2l[1]
                    else:
                        n2[2] ^ self._front[index[0]]
                        n1[1] ^ self._front[index[1]]
                        self._nodes.append(n1)
                        self._nodes.append(n2)
                        self._front[index[0]] = n1[0]
                        self._front[index[1]] = n2[1]
                        if self.is_dm:
                            [n1l, n2l], _ = self.copy_nodes([n1, n2], conj=True)
                            n2l[1] ^ self._front[index[0] + nq]
                            n1l[2] ^ self._front[index[1] + nq]
                            self._nodes.append(n1l)
                            self._nodes.append(n2l)
                            self._front[index[0] + nq] = n1l[0]
                            self._front[index[1] + nq] = n2l[1]
                    applied = True

            if applied is False:
                gate.name = name
                self.coloring_nodes([gate], flag="gate")
                # gate.flag = "gate"
                # gate.is_dagger = False
                # gate.id = id(gate)
                self._nodes.append(gate)
                if self.is_dm:
                    lgates, _ = self.copy_nodes([gate], conj=True)
                    lgate = lgates[0]
                    self._nodes.append(lgate)
                for i, ind in enumerate(index):
                    gate.get_edge(i + noe) ^ self._front[ind]
                    self._front[ind] = gate.get_edge(i)
                    if self.is_dm:
                        lgate.get_edge(i + noe) ^ self._front[ind + nq]
                        self._front[ind + nq] = lgate.get_edge(i)

        elif mpo:  # gate in MPO format
            assert isinstance(gate, QuOperator)
            gatec = gate.copy()
            self.coloring_nodes(gatec.nodes, flag="gate")
            for n in gatec.nodes:
                n.id = id(gate)
                n.name = name
            self._nodes += gatec.nodes
            if self.is_dm:
                gateconj = gate.adjoint()
                self.coloring_nodes(gateconj.nodes, flag="gate", is_dagger=True)
                for _, n in zip(gatec.nodes, gateconj.nodes):
                    n.id = id(gate)
                    n.name = name
                self._nodes += gateconj.nodes

            for i, ind in enumerate(index):
                gatec.in_edges[i] ^ self._front[ind]
                self._front[ind] = gatec.out_edges[i]
                if self.is_dm:
                    gateconj.out_edges[i] ^ self._front[ind + nq]
                    self._front[ind + nq] = gateconj.in_edges[i]

        elif diagonal:
            if isinstance(gate, tn.Node):
                mps_nodes = [gate]
            else:
                mps_nodes = gate.nodes
            self.coloring_nodes(mps_nodes, flag="gate")
            for n in mps_nodes:
                n.id = id(gate)
                n.name = name
            self._nodes += mps_nodes

            if self.is_dm:
                if isinstance(gate, tn.Node):
                    gateconj_tensor = backend.conj(gate.tensor)
                    gateconj_node = tn.Node(gateconj_tensor, name=name)
                    gateconj_nodes = [gateconj_node]
                else:
                    gateconj = gate.adjoint()
                    gateconj_nodes = gateconj.nodes
                self.coloring_nodes(gateconj_nodes, flag="gate", is_dagger=True)
                for _, n in zip(mps_nodes, gateconj_nodes):
                    n.id = id(gate)
                    n.name = name
                self._nodes += gateconj_nodes

            for i, ind in enumerate(index):
                if isinstance(gate, tn.Node):
                    phys_edge = gate[i]
                else:
                    phys_edge = gate.out_edges[i]

                cn = tn.CopyNode(3, self._d, name=f"{name}_copy_{i}")
                self.coloring_nodes([cn], flag="gate")
                self._nodes.append(cn)

                cn[0] ^ self._front[ind]
                cn[1] ^ phys_edge
                self._front[ind] = cn[2]

                if self.is_dm:
                    if isinstance(gate, tn.Node):
                        phys_edge_conj = gateconj_nodes[0][i]
                    else:
                        phys_edge_conj = gateconj.in_edges[i]

                    cn_conj = tn.CopyNode(3, self._d, name=f"{name}_copy_{i}_conj")
                    self.coloring_nodes([cn_conj], flag="gate", is_dagger=True)
                    self._nodes.append(cn_conj)

                    cn_conj[0] ^ self._front[ind + nq]
                    cn_conj[1] ^ phys_edge_conj
                    self._front[ind + nq] = cn_conj[2]

        self.state_tensor = None  # refresh the state cache

    apply = apply_general_gate

    def _copy_state_tensor(
        self, conj: bool = False, reuse: bool = True
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        if reuse:
            t = getattr(self, "state_tensor", None)
            if t is None:
                nodes, d_edges = self._copy()
                t = contractor(nodes, output_edge_order=d_edges)
                setattr(self, "state_tensor", t)
            ndict, edict = tn.copy([t], conjugate=conj)
            newnodes = []
            newnodes.append(ndict[t])
            newfront = []
            for e in t.edges:
                newfront.append(edict[e])
            return newnodes, newfront
        return self._copy(conj)

    def expectation_before(
        self,
        *ops: Tuple[tn.Node, List[int]],
        reuse: bool = True,
        **kws: Any,
    ) -> List[tn.Node]:
        """
        Get the tensor network in the form of a list of nodes
        for the expectation calculation before the real contraction

        :param reuse: _description_, defaults to True
        :type reuse: bool, optional
        :raises ValueError: _description_
        :return: _description_
        :rtype: List[tn.Node]
        """
        nq = self._nqubits
        if self.is_dm is True:
            nodes, newdang = self._copy_state_tensor(reuse=reuse)
        else:
            nodes1, edge1 = self._copy_state_tensor(reuse=reuse)
            nodes2, edge2 = self._copy_state_tensor(conj=True, reuse=reuse)
            nodes = nodes1 + nodes2
            newdang = edge1 + edge2
        occupied = set()
        for op, index in ops:
            if not isinstance(op, tn.Node):
                # op is only a matrix
                op = backend.reshaped(op, d=self._d)
                op = backend.cast(op, dtype=dtypestr)
                op = gates.Gate(op)
            else:
                op.tensor = backend.cast(op.tensor, dtype=dtypestr)
            if isinstance(index, int):
                index = [index]
            index = tuple(i if i >= 0 else self._nqubits + i for i in index)  # type: ignore
            noe = len(index)

            for j, e in enumerate(index):
                if e in occupied:
                    raise ValueError("Cannot measure two operators in one index")
                newdang[e + nq] ^ op.get_edge(j)
                newdang[e] ^ op.get_edge(j + noe)
                occupied.add(e)
            self.coloring_nodes([op], flag="operator")
            # op.flag = "operator"
            # op.is_dagger = False
            # op.id = id(op)
            nodes.append(op)
        for j in range(nq):
            if j not in occupied:  # edge1[j].is_dangling invalid here!
                newdang[j] ^ newdang[j + nq]
        return nodes

    def perfect_sampling(self, status: Optional[Tensor] = None) -> Tuple[str, float]:
        """
        Sampling base-d strings (0-9A-Z when d <= 36) from the circuit output based on quantum amplitudes.
        Reference: arXiv:1201.3974.

        :param status: external randomness, with shape [nqubits], defaults to None
        :type status: Optional[Tensor]
        :return: Sampled base-d string and the corresponding theoretical probability.
        :rtype: Tuple[str, float]
        """
        return self.measure_jit(*range(self._nqubits), with_prob=True, status=status)

    def measure_jit(
        self, *index: int, with_prob: bool = False, status: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Take measurement on the given site indices (computational basis).
        This method is jittable is and about 100 times faster than unjit version!

        :param index: Measure on which site (wire) index.
        :type index: int
        :param with_prob: If true, theoretical probability is also returned.
        :type with_prob: bool, optional
        :param status: external randomness, with shape [index], defaults to None
        :type status: Optional[Tensor]
        :return: The sample output and probability (optional) of the quantum line.
        :rtype: Tuple[Tensor, Tensor]
        """
        # finally jit compatible ! and much faster than unjit version ! (100x)
        sample: List[Tensor] = []
        one_r = backend.cast(backend.convert_to_tensor(1.0), rdtypestr)
        p = one_r
        for k, j in enumerate(index):
            if self.is_dm is False:
                nodes1, edge1 = self._copy()
                nodes2, edge2 = self._copy(conj=True)
                newnodes = nodes1 + nodes2
            else:
                newnodes, newfront = self._copy()
                nfront = len(newfront) // 2
                edge2 = newfront[nfront:]
                edge1 = newfront[:nfront]
            for i, e in enumerate(edge1):
                if i != j:
                    e ^ edge2[i]
            for i in range(k):
                if self._d == 2:
                    m = (1 - sample[i]) * gates.array_to_tensor(
                        np.array([1, 0])
                    ) + sample[i] * gates.array_to_tensor(np.array([0, 1]))
                else:
                    m = onehot_d_tensor(sample[i], d=self._d)
                g1 = Gate(m)
                self.coloring_nodes([g1], flag="measurement")
                newnodes.append(g1)
                g1.get_edge(0) ^ edge1[index[i]]
                g2 = Gate(m)
                self.coloring_nodes([g2], flag="measurement", is_dagger=True)
                newnodes.append(g2)
                g2.get_edge(0) ^ edge2[index[i]]

            rho = (
                1
                / backend.cast(p, dtypestr)
                * contractor(newnodes, output_edge_order=[edge1[j], edge2[j]]).tensor
            )
            if self._d == 2:
                pu = backend.real(rho[0, 0])
                if status is None:
                    r = backend.implicit_randu()[0]
                else:
                    r = status[k]
                r = backend.real(backend.cast(r, dtypestr))
                eps = 0.31415926 * 1e-12
                sign = (
                    backend.sign(r - pu + eps) / 2 + 0.5
                )  # in case status is exactly 0.5
                sign = backend.convert_to_tensor(sign)
                sign = backend.cast(sign, dtype=rdtypestr)
                sign_complex = backend.cast(sign, dtypestr)
                sample.append(sign_complex)
                p = p * (pu * (-1) ** sign + sign)
            else:
                pu = backend.clip(
                    backend.real(backend.diagonal(rho)),
                    backend.convert_to_tensor(0.0),
                    backend.convert_to_tensor(1.0),
                )
                pu = pu / backend.sum(pu)
                if status is None:
                    ind = backend.implicit_randc(
                        a=backend.arange(self._d),
                        shape=1,
                        p=backend.cast(pu, rdtypestr),
                    )
                else:
                    one_r = backend.cast(backend.convert_to_tensor(1.0), rdtypestr)
                    st = backend.cast(status[k : k + 1], rdtypestr)
                    ind = backend.probability_sample(
                        shots=1,
                        p=backend.cast(pu, rdtypestr),
                        status=one_r - st,
                    )
                k_out = backend.cast(ind[0], "int32")
                sample.append(backend.cast(k_out, rdtypestr))
                p = p * backend.cast(pu[k_out], rdtypestr)
        sample = backend.real(backend.stack(sample))
        if with_prob:
            return sample, p
        else:
            return sample, -1.0

    measure = measure_jit

    def amplitude_before(self, l: Union[str, Tensor]) -> List[Gate]:
        r"""
        Returns the tensornetwor nodes for the amplitude of the circuit given the bitstring l.
        For state simulator, it computes :math:`\langle l\vert \psi\rangle`,
        for density matrix simulator, it computes :math:`Tr(\rho \vert l\rangle \langle 1\vert)`
        Note how these two are different up to a square operation.

        :param l: The bitstring of 0 and 1s.
        :type l: Union[str, Tensor]
        :return: The tensornetwork nodes for the amplitude of the circuit.
        :rtype: List[Gate]
        """

        no, d_edges = self._copy()
        if isinstance(l, str):
            l = _decode_basis_label(l, n=self._nqubits, dim=self._d)
        l = backend.convert_to_tensor(l)
        endns = onehot_d_tensor(l, d=self._d)
        ms = []
        if self.is_dm:
            msconj = []
        for i in range(self._nqubits):
            n = tn.Node(endns[i])
            self.coloring_nodes([n], flag="measurement")
            ms.append(n)
            d_edges[i] ^ n.get_edge(0)
            if self.is_dm:
                nconj = tn.Node(endns[i])
                self.coloring_copied_nodes(
                    [nconj], [n], flag="measurement", is_dagger=True
                )
                msconj.append(nconj)
                d_edges[i + self._nqubits] ^ nconj.get_edge(0)
        no.extend(ms)
        if self.is_dm:
            no.extend(msconj)
        return no

    def amplitude(self, l: Union[str, Tensor]) -> Tensor:
        r"""
        Returns the amplitude of the circuit given the bitstring l.
        For state simulator, it computes :math:`\langle l\vert \psi\rangle`,
        for density matrix simulator, it computes :math:`Tr(\rho \vert l\rangle \langle 1\vert)`
        Note how these two are different up to a square operation.

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.X(0)
        >>> c.amplitude("10")
        array(1.+0.j, dtype=complex64)
        >>> c.CNOT(0, 1)
        >>> c.amplitude("11")
        array(1.+0.j, dtype=complex64)

        :param l: The bitstring of 0 and 1s.
        :type l: Union[str, Tensor]
        :return: The amplitude of the circuit.
        :rtype: tn.Node.tensor
        """
        no = self.amplitude_before(l)

        return contractor(no).tensor

    def probability(self) -> Tensor:
        """
        get the d^n length probability vector over computational basis

        :return: probability vector of shape [dim**n]
        :rtype: Tensor
        """
        s = self.state()  # type: ignore
        if self.is_dm is False:
            amp = backend.reshape(s, [-1])
            p = backend.real(backend.abs(amp) ** 2)
        else:
            diag = backend.diagonal(s)
            p = backend.real(backend.reshape(diag, [-1]))
        return p

    def _merge_qir_with_extra(self) -> List[Dict[str, Any]]:
        from .translation import _merge_extra_qir

        return _merge_extra_qir(self._qir, getattr(self, "_extra_qir", []))

    @staticmethod
    def _is_measure_instruction_name(name: str) -> bool:
        return name.upper() in ["MEASURE", "M", "MZ", "MR"]

    @staticmethod
    def _is_random_instruction_name(name: str) -> bool:
        return name.upper() in [
            "MEASURE",
            "M",
            "MZ",
            "MR",
            "RESET",
            "DEPOLARIZING",
            "PAULI",
        ]

    def _count_detector_random_events(
        self, merged_qir: Sequence[Dict[str, Any]]
    ) -> int:
        c = 0
        for d in merged_qir:
            if d.get("is_channel", False):
                if self._is_random_instruction_name(str(d.get("name", ""))):
                    c += 1
                continue
            if d.get("instruction", False) and self._is_random_instruction_name(
                str(d.get("name", ""))
            ):
                c += 1
        return c

    def _resolve_detectors(
        self, merged_qir: Sequence[Dict[str, Any]]
    ) -> List[List[int]]:
        measured_so_far = 0
        resolved: List[List[int]] = []
        for d in merged_qir:
            if not d.get("instruction", False):
                continue
            name = str(d.get("name", "")).upper()
            if self._is_measure_instruction_name(name):
                measured_so_far += 1
                continue
            if name != "DETECTOR":
                continue
            lookback_indices = [int(v) for v in d.get("index", [])]
            base = int(d.get("current_m_count", measured_so_far))
            abs_indices = []
            for v in lookback_indices:
                rid = base + v if v < 0 else v
                if rid < 0:
                    raise ValueError(
                        f"Invalid detector lookback index {v} with base {base}."
                    )
                if rid >= measured_so_far:
                    raise ValueError(
                        f"Detector index {rid} is out of range for {measured_so_far} measurements."
                    )
                abs_indices.append(rid)
            resolved.append(abs_indices)
        return resolved

    def _record_qubits_terminal_measure_only(
        self, merged_qir: Sequence[Dict[str, Any]]
    ) -> List[int]:
        measured_so_far = 0
        record_to_qubit: Dict[int, int] = {}
        for d in merged_qir:
            if not d.get("instruction", False):
                continue
            name = str(d.get("name", "")).upper()
            if name in ["DETECTOR", "BARRIER", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"]:
                continue
            if name in ["MEASURE", "M", "MZ"]:
                if int(d.get("pos", len(self._qir))) < len(self._qir):
                    raise NotImplementedError(
                        "allow_state=True currently supports terminal measure_instruction only."
                    )
                rid = int(d.get("record_index", measured_so_far))
                record_to_qubit[rid] = int(d["index"][0])
                measured_so_far += 1
                continue
            raise NotImplementedError(
                f"allow_state=True does not support instruction `{name}`."
            )
        if measured_so_far == 0:
            raise ValueError("No measurements defined for detector sampling.")
        record_qubits = [-1] * measured_so_far
        for rid, q in record_to_qubit.items():
            if rid < measured_so_far:
                record_qubits[rid] = q
        if any(q < 0 for q in record_qubits):
            raise ValueError(
                "Measurement records are incomplete for detector resolution."
            )
        return record_qubits

    def _run_instruction_trajectory(
        self,
        merged_qir: Sequence[Dict[str, Any]],
        status_row: Optional[Tensor] = None,
    ) -> Tensor:
        c = type(self)(**self.circuit_param)
        active = [True] * self._nqubits
        records: List[Tensor] = []
        status_i = 0
        status_len = 0
        if status_row is not None:
            status_len = int(backend.shape_tuple(status_row)[0])

        def next_status() -> Optional[Tensor]:
            nonlocal status_i
            if status_row is None:
                return None
            if status_i >= status_len:
                raise ValueError(
                    "Provided status is shorter than required random events."
                )
            s = status_row[status_i]
            status_i += 1
            return s

        for d in merged_qir:
            name = str(d.get("name", "")).upper()
            if d.get("is_channel", False):
                q = int(d["index"][0])
                if not active[q]:
                    raise NotImplementedError(
                        "Noise after measure requires reset first."
                    )
                if name in ["DEPOLARIZING", "PAULI"]:
                    params = d.get("channel_parameters", {})
                    px = float(params.get("px", 0.0))
                    py = float(params.get("py", 0.0))
                    pz = float(params.get("pz", 0.0))
                    c.depolarizing(q, px=px, py=py, pz=pz, status=next_status())  # type: ignore
                    continue
                self._apply_qir(c, [d])  # type: ignore[arg-type]
                continue
            if not d.get("instruction", False):
                inds = [int(i) for i in d.get("index", [])]
                for q in inds:
                    if q < self._nqubits and not active[q]:
                        raise NotImplementedError(
                            "Gate after measure requires reset_instruction first."
                        )
                self._apply_qir(c, [d])  # type: ignore[arg-type]
                continue
            if name in ["MEASURE", "M", "MZ"]:
                q = int(d["index"][0])
                if not active[q]:
                    raise NotImplementedError("Repeated measure on inactive line.")
                m = backend.cast(c.cond_measurement(q, status=next_status()), "int32")
                records.append(m)
                active[q] = False
                continue
            if name == "MR":
                q = int(d["index"][0])
                if not active[q]:
                    raise NotImplementedError("MR on inactive line.")
                m = backend.cast(c.cond_measurement(q, status=next_status()), "int32")
                records.append(m)
                c.conditional_gate(m, [gates.i(), gates.x()], q)  # type: ignore
                active[q] = True
                continue
            if name == "RESET":
                q = int(d["index"][0])
                m = c.cond_measurement(q, status=next_status())
                c.conditional_gate(m, [gates.i(), gates.x()], q)  # type: ignore
                active[q] = True
                continue
            if name in ["DEPOLARIZING", "PAULI"]:
                q = int(d["index"][0])
                params = d.get("parameters", {})
                px = float(params.get("px", d.get("px", 0.0)))
                py = float(params.get("py", d.get("py", 0.0)))
                pz = float(params.get("pz", d.get("pz", 0.0)))
                c.depolarizing(q, px=px, py=py, pz=pz, status=next_status())  # type: ignore
                continue
            if name in ["DETECTOR", "BARRIER", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"]:
                continue
            raise NotImplementedError(
                f"Unsupported instruction in sample_detector: {name}"
            )
        if status_row is not None and status_i != status_len:
            logger.warning("Provided status is longer than required random events.")
        if len(records) == 0:
            return backend.cast(backend.convert_to_tensor([]), "int32")
        return backend.cast(backend.stack(records), "int32")

    def _build_detector_tn_wht(
        self, record_qubits: Sequence[int], resolved_detectors: Sequence[Sequence[int]]
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=npdtype)
        tracev = np.array([1.0, 1.0], dtype=npdtype)
        if self.is_dm:
            nodes, front = self._copy()
            ket_front = front[: self._nqubits]
            bra_front = front[self._nqubits : 2 * self._nqubits]
            nodes = list(nodes)
        else:
            ket_nodes, ket_front = self._copy()
            bra_nodes, bra_front = self._copy(conj=True)
            nodes = list(ket_nodes) + list(bra_nodes)

        record_edges: List[Optional[tn.Edge]] = [None] * len(record_qubits)
        qubit_records: Dict[int, List[int]] = {}
        for rid, q in enumerate(record_qubits):
            qubit_records.setdefault(q, []).append(rid)

        for q in range(self._nqubits):
            rids = qubit_records.get(q, [])
            if len(rids) == 0:
                ket_front[q] ^ bra_front[q]
                continue
            if len(rids) > 1:
                raise NotImplementedError(
                    "allow_state=True currently supports at most one terminal measurement per qubit."
                )
            qcopy = tn.CopyNode(3, 2)
            qcopy[0] ^ ket_front[q]
            qcopy[1] ^ bra_front[q]
            nodes.append(qcopy)
            record_edges[rids[0]] = qcopy[2]

        rid_usage_count: Dict[int, int] = {}
        for det in resolved_detectors:
            for rid in det:
                rid_usage_count[rid] = rid_usage_count.get(rid, 0) + 1

        fanned_out_edges: Dict[int, List[tn.Edge]] = {}
        for rid, count in rid_usage_count.items():
            redge = record_edges[rid]
            if redge is None:
                raise ValueError(f"Missing measurement edge for record {rid}.")
            if count == 1:
                fanned_out_edges[rid] = [redge]
            else:
                fanout = tn.CopyNode(count + 1, 2)
                nodes.append(fanout)
                redge ^ fanout[0]
                fanned_out_edges[rid] = [fanout[i + 1] for i in range(count)]

        rid_pop_index = {rid: 0 for rid in rid_usage_count}
        detector_edges: List[tn.Edge] = []
        for det in resolved_detectors:
            dcopy = tn.CopyNode(len(det) + 1, 2)
            nodes.append(dcopy)
            for i, rid in enumerate(det):
                hnode = tn.Node(hadamard)
                nodes.append(hnode)
                edge_to_use = fanned_out_edges[rid][rid_pop_index[rid]]
                rid_pop_index[rid] += 1
                edge_to_use ^ hnode[0]
                hnode[1] ^ dcopy[i]
            hout = tn.Node(hadamard)
            nodes.append(hout)
            dcopy[len(det)] ^ hout[0]
            detector_edges.append(hout[1])

        for rid, redge in enumerate(record_edges):
            if rid in rid_usage_count:
                continue
            if redge is None:
                raise ValueError(f"Missing measurement edge for record {rid}.")
            tnode = tn.Node(tracev)
            nodes.append(tnode)
            redge ^ tnode[0]
        return nodes, detector_edges

    def detector_probabilities(self) -> Tensor:
        """
        Calculate the joint probability distribution of all detectors in the circuit.

        :return: A tensor representing the joint probability distribution.
        :rtype: Tensor
        """
        merged_qir = self._merge_qir_with_extra()
        resolved_detectors = self._resolve_detectors(merged_qir)
        if len(resolved_detectors) == 0:
            raise ValueError("No detectors defined in the circuit.")
        nodes, detector_edges = self._build_detector_tn_from_qir(
            merged_qir, resolved_detectors
        )
        pt = contractor(nodes, output_edge_order=detector_edges).tensor
        p = backend.abs(backend.real(pt))
        return p / backend.sum(p)

    def outcome_probability(self, state: Sequence[int]) -> Tensor:
        """
        Calculate the probability of a specific detector outcome bitstring.

        :param state: The detector outcome bitstring as a sequence of 0s and 1s.
        :type state: Sequence[int]
        :return: The probability of the given outcome.
        :rtype: Tensor
        """
        merged_qir = self._merge_qir_with_extra()
        resolved_detectors = self._resolve_detectors(merged_qir)
        if len(resolved_detectors) == 0:
            raise ValueError("No detectors defined in the circuit.")
        if len(state) != len(resolved_detectors):
            raise ValueError(
                f"State length {len(state)} does not match number of detectors {len(resolved_detectors)}."
            )
        nodes, detector_edges = self._build_detector_tn_from_qir(
            merged_qir, resolved_detectors
        )
        nodes_num, detector_edges_num = self.copy_nodes(nodes, detector_edges)
        for i, s in enumerate(state):
            s_val = backend.cast(backend.convert_to_tensor(s), "int32")
            p_vec = backend.onehot(s_val, 2)
            p_node = tn.Node(backend.cast(p_vec, dtypestr))
            nodes_num.append(p_node)
            detector_edges_num[i] ^ p_node[0]
        p_num = contractor(nodes_num).tensor

        nodes_den, detector_edges_den = self.copy_nodes(nodes, detector_edges)
        tracev = tn.Node(backend.cast(backend.convert_to_tensor([1.0, 1.0]), dtypestr))
        for i in range(len(detector_edges_den)):
            tnode = tn.Node(tracev.tensor)
            nodes_den.append(tnode)
            detector_edges_den[i] ^ tnode[0]
        p_den = contractor(nodes_den).tensor
        return backend.abs(backend.real(p_num)) / backend.abs(backend.real(p_den))

    def _build_detector_tn_from_qir(
        self,
        merged_qir: Sequence[Dict[str, Any]],
        resolved_detectors: Sequence[Sequence[int]],
    ) -> Tuple[List[tn.Node], List[tn.Edge]]:
        if self._can_build_detector_from_existing_nodes(merged_qir):
            record_qubits = self._record_qubits_terminal_measure_only(merged_qir)
            return self._build_detector_tn_wht(record_qubits, resolved_detectors)

        n = self._nqubits
        work = self._new_detector_work_circuit()
        work._qir = []
        work._extra_qir = []
        active = [True] * n
        record_edges: Dict[int, tn.Edge] = {}
        measured_so_far = 0
        hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=npdtype)
        tracev = np.array([1.0, 1.0], dtype=npdtype)
        zero = np.array([1.0, 0.0], dtype=npdtype)

        for d in merged_qir:
            name = str(d.get("name", "")).upper()
            if not d.get("instruction", False):
                inds = [int(i) for i in d.get("index", [])]
                for q in inds:
                    if q < n and not active[q]:
                        raise NotImplementedError(
                            "Gate/noise after measure requires reset_instruction first."
                        )
                self._apply_qir(work, [d])  # type: ignore[arg-type]
                continue

            if name in ["MEASURE", "M", "MZ"]:
                q = int(d["index"][0])
                if not active[q]:
                    raise NotImplementedError("Repeated measure on inactive line.")
                qcopy = tn.CopyNode(3, 2)
                work._nodes.append(qcopy)
                qcopy[0] ^ work._front[q]
                qcopy[1] ^ work._front[q + n]
                rid = int(d.get("record_index", measured_so_far))
                measured_so_far += 1
                record_edges[rid] = qcopy[2]
                active[q] = False
                continue

            if name == "MR":
                q = int(d["index"][0])
                if not active[q]:
                    raise NotImplementedError("MR on inactive line.")
                qcopy = tn.CopyNode(3, 2)
                work._nodes.append(qcopy)
                qcopy[0] ^ work._front[q]
                qcopy[1] ^ work._front[q + n]
                rid = int(d.get("record_index", measured_so_far))
                measured_so_far += 1
                record_edges[rid] = qcopy[2]
                kn = tn.Node(zero)
                bn = tn.Node(zero)
                work._nodes.extend([kn, bn])
                work._front[q] = kn[0]
                work._front[q + n] = bn[0]
                active[q] = True
                continue

            if name == "RESET":
                q = int(d["index"][0])
                if active[q]:
                    work._front[q] ^ work._front[q + n]
                kn = tn.Node(zero)
                bn = tn.Node(zero)
                work._nodes.extend([kn, bn])
                work._front[q] = kn[0]
                work._front[q + n] = bn[0]
                active[q] = True
                continue

            if name in ["DETECTOR", "BARRIER", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"]:
                continue

            if name in ["DEPOLARIZING", "PAULI"]:
                q = int(d["index"][0])
                if not active[q]:
                    raise NotImplementedError("Noise on inactive line.")
                params = d.get("parameters", {})
                px = float(params.get("px", d.get("px", 0.0)))
                py = float(params.get("py", d.get("py", 0.0)))
                pz = float(params.get("pz", d.get("pz", 0.0)))
                work.depolarizing(q, px=px, py=py, pz=pz)  # type: ignore
                continue

            raise NotImplementedError(
                f"Unsupported instruction `{name}` in detector TN."
            )

        for q in range(n):
            if active[q]:
                work._front[q] ^ work._front[q + n]

        nodes = list(work._nodes)
        rid_usage_count: Dict[int, int] = {}
        for det in resolved_detectors:
            for rid in det:
                rid_usage_count[rid] = rid_usage_count.get(rid, 0) + 1

        fanned_out_edges: Dict[int, List[tn.Edge]] = {}
        for rid, count in rid_usage_count.items():
            if rid not in record_edges:
                raise ValueError(
                    f"Detector references unknown measurement record {rid}."
                )
            redge = record_edges[rid]
            if count == 1:
                fanned_out_edges[rid] = [redge]
            else:
                fanout = tn.CopyNode(count + 1, 2)
                nodes.append(fanout)
                redge ^ fanout[0]
                fanned_out_edges[rid] = [fanout[i + 1] for i in range(count)]

        rid_pop_index = {rid: 0 for rid in rid_usage_count}
        detector_edges: List[tn.Edge] = []
        for det in resolved_detectors:
            dcopy = tn.CopyNode(len(det) + 1, 2)
            nodes.append(dcopy)
            for i, rid in enumerate(det):
                hnode = tn.Node(hadamard)
                nodes.append(hnode)
                edge_to_use = fanned_out_edges[rid][rid_pop_index[rid]]
                rid_pop_index[rid] += 1
                edge_to_use ^ hnode[0]
                hnode[1] ^ dcopy[i]
            hout = tn.Node(hadamard)
            nodes.append(hout)
            dcopy[len(det)] ^ hout[0]
            detector_edges.append(hout[1])

        for rid, redge in record_edges.items():
            if rid in rid_usage_count:
                continue
            tnode = tn.Node(tracev)
            nodes.append(tnode)
            redge ^ tnode[0]
        return nodes, detector_edges

    def _new_detector_work_circuit(self) -> "BaseCircuit":
        return type(self)(**self.circuit_param)

    def _can_build_detector_from_existing_nodes(
        self, merged_qir: Sequence[Dict[str, Any]]
    ) -> bool:
        for d in merged_qir:
            if d.get("is_channel", False):
                return False
            if not d.get("instruction", False):
                continue
            name = str(d.get("name", "")).upper()
            if name in ["DETECTOR", "BARRIER", "TICK", "QUBIT_COORDS", "SHIFT_COORDS"]:
                continue
            if name in ["MEASURE", "M", "MZ"]:
                if int(d.get("pos", len(self._qir))) < len(self._qir):
                    return False
                continue
            return False
        return True

    def sample_detector(
        self,
        shots: int = 1,
        batch: Optional[int] = None,
        allow_state: bool = False,
        status: Optional[Tensor] = None,
        seed: Optional[int] = None,
        **kws: Any,
    ) -> Tensor:
        """
        Sample detector outcomes from instruction-annotated circuits.

        :param shots: Number of samples to draw, defaults to 1.
        :type shots: int, optional
        :param batch: Number of samples to process in a single batch, defaults to None (equal to shots).
        :type batch: int, optional
        :param allow_state: If True, uses the full detector probability distribution for sampling
            (faster but memory-intensive); if False, uses an autoregressive sampling method based on
            the tensor network, defaults to False.
        :type allow_state: bool, optional
        :param status: Random numbers in [0, 1] used for sampling, defaults to None.
            If allow_state is True, shape should be [shots] or [shots, 1];
            if allow_state is False, shape should be [shots, num_detectors].
        :type status: Optional[Tensor], optional
        :param seed: Random seed for sampling, defaults to None.
        :type seed: Optional[int], optional
        :return: A boolean tensor containing the sampled detector outcomes with shape [shots, num_detectors].
        :rtype: Tensor
        """
        if "batch_size" in kws and "shots" not in kws:
            shots = int(kws["batch_size"])
        if "shots" in kws:
            shots = int(kws["shots"])
        if shots <= 0:
            raise ValueError("shots must be positive.")
        if batch is None:
            batch = shots
        if batch <= 0:
            raise ValueError("batch must be positive.")

        merged_qir = self._merge_qir_with_extra()
        resolved_detectors = self._resolve_detectors(merged_qir)
        if len(resolved_detectors) == 0:
            raise ValueError("No detectors defined in the circuit.")
        num_detectors = len(resolved_detectors)

        if allow_state:
            status_state: Optional[Tensor] = None
            if status is not None:
                status_state = backend.cast(
                    backend.convert_to_tensor(status), rdtypestr
                )
                ss = backend.shape_tuple(status_state)
                if len(ss) == 2:
                    if int(ss[1]) != 1:
                        raise ValueError(
                            "allow_state=True expects status shape [shots] or [shots, 1]."
                        )
                    status_state = backend.reshape(status_state, [int(ss[0])])
                    ss = backend.shape_tuple(status_state)
                if len(ss) != 1 or int(ss[0]) != shots:
                    raise ValueError(
                        "allow_state=True expects status shape [shots] or [shots, 1]."
                    )
            else:
                if seed is not None:
                    g = backend.get_random_state(seed)
                    status_state = backend.cast(
                        backend.stateful_randu(g, shape=[shots]), rdtypestr
                    )
            p = backend.reshape(self.detector_probabilities(), [-1])
            sample_int = backend.probability_sample(shots, p, status_state)
            sample_bin = sample_int2bin(sample_int, num_detectors, dim=2)
            return backend.cast(sample_bin, "bool")

        status2d: Optional[Tensor]
        if (not self.is_dm) and (not allow_state):
            num_random_events = self._count_detector_random_events(merged_qir)
            if status is not None:
                status2d = backend.cast(backend.convert_to_tensor(status), rdtypestr)
                ss = backend.shape_tuple(status2d)
                if len(ss) == 1:
                    if shots != 1:
                        raise ValueError("1D status is only valid when shots == 1.")
                    status2d = backend.reshape(status2d, [1, int(ss[0])])
                    ss = backend.shape_tuple(status2d)
                if len(ss) != 2 or int(ss[0]) != shots:
                    raise ValueError(
                        "allow_state=False trajectory mode expects status shape [shots, num_random_events]."
                    )
                if int(ss[1]) != num_random_events:
                    raise ValueError(
                        f"status second dimension must equal number of random events ({num_random_events})."
                    )
            else:
                if num_random_events == 0:
                    status2d = None
                elif seed is not None:
                    g = backend.get_random_state(seed)
                    status2d = backend.cast(
                        backend.stateful_randu(g, shape=[shots, num_random_events]),
                        rdtypestr,
                    )
                else:
                    status2d = backend.cast(
                        backend.implicit_randu(shape=[shots, num_random_events]),
                        rdtypestr,
                    )

            two_i = backend.cast(backend.convert_to_tensor(2), "int32")

            def trajectory_one(row: Any) -> Tensor:
                records = self._run_instruction_trajectory(merged_qir, row)
                det_row: List[Tensor] = []
                for rec_indices in resolved_detectors:
                    inds = backend.cast(
                        backend.convert_to_tensor(rec_indices), dtype="int32"
                    )
                    bits = backend.gather1d(records, inds)
                    parity = backend.mod(backend.sum(bits), two_i)
                    det_row.append(backend.cast(parity, "int32"))
                return backend.stack(det_row)

            trajectory_batch = backend.jit(backend.vmap(trajectory_one))

            if status2d is None:
                # num_random_events == 0 case, results are deterministic
                res = backend.jit(trajectory_one)(None)
                return backend.cast(backend.tile(res[None, :], [shots, 1]), "bool")

            shot_rows: List[Tensor] = []
            start = 0
            while start < shots:
                stop = min(start + batch, shots)
                sb = status2d[start:stop]
                shot_rows.append(trajectory_batch(sb))
                start = stop
            return backend.cast(backend.concat(shot_rows, axis=0), "bool")

        if status is not None:
            status2d = backend.cast(backend.convert_to_tensor(status), rdtypestr)
            ss = backend.shape_tuple(status2d)
            if len(ss) == 1:
                if shots != 1:
                    raise ValueError("1D status is only valid when shots == 1.")
                status2d = backend.reshape(status2d, [1, int(ss[0])])
                ss = backend.shape_tuple(status2d)
            if len(ss) != 2 or int(ss[0]) != shots:
                raise ValueError(
                    "allow_state=False expects status shape [shots, num_detectors]."
                )
            if int(ss[1]) != num_detectors:
                raise ValueError(
                    f"status second dimension must equal number of detectors ({num_detectors})."
                )
        else:
            if seed is not None:
                g = backend.get_random_state(seed)
                status2d = backend.cast(
                    backend.stateful_randu(g, shape=[shots, num_detectors]), rdtypestr
                )
            else:
                status2d = backend.cast(
                    backend.implicit_randu(shape=[shots, num_detectors]), rdtypestr
                )

        base_nodes, base_detector_edges = self._build_detector_tn_from_qir(
            merged_qir, resolved_detectors
        )
        tracev = backend.cast(backend.convert_to_tensor([1.0, 1.0]), dtypestr)

        def sample_one(status_row: Tensor) -> Tensor:
            assigned: List[Tensor] = []
            for j in range(num_detectors):
                nodes, detector_edges = self.copy_nodes(base_nodes, base_detector_edges)
                for k, bit in enumerate(assigned):
                    pnode = tn.Node(backend.cast(backend.onehot(bit, 2), dtypestr))
                    nodes.append(pnode)
                    detector_edges[k] ^ pnode[0]
                for k in range(j + 1, num_detectors):
                    tnode = tn.Node(tracev)
                    nodes.append(tnode)
                    detector_edges[k] ^ tnode[0]
                pt = contractor(nodes, output_edge_order=[detector_edges[j]]).tensor
                p2 = backend.abs(backend.real(backend.reshape(pt, [2])))
                p2 = p2 / backend.sum(p2)
                sampled = backend.probability_sample(1, p2, status_row[j : j + 1])[0]
                assigned.append(backend.cast(sampled, "int32"))
            return backend.cast(backend.stack(assigned), "bool")

        sample_one_jit = backend.jit(sample_one)
        sample_batch_jit = backend.jit(backend.vmap(sample_one_jit))

        outs = []
        start = 0
        while start < shots:
            stop = min(start + batch, shots)
            sb = status2d[start:stop]
            outs.append(sample_batch_jit(sb))
            start = stop
        if len(outs) == 1:
            return outs[0]
        return backend.concat(outs, axis=0)

    @partial(arg_alias, alias_dict={"format": ["format_"]})
    def sample(
        self,
        batch: Optional[int] = None,
        allow_state: bool = False,
        readout_error: Optional[Sequence[Any]] = None,
        format: Optional[str] = None,
        random_generator: Optional[Any] = None,
        status: Optional[Tensor] = None,
        jittable: bool = True,
    ) -> Any:
        r"""
        batched sampling from state or circuit tensor network directly

        :param batch: number of samples, defaults to None
        :type batch: Optional[int], optional
        :param allow_state: if true, we sample from the final state
            if memory allows, True is preferred, defaults to False
        :type allow_state: bool, optional
        :param readout_error: readout_error, defaults to None
        :type readout_error: Optional[Sequence[Any]]. Tensor, List, Tuple
        :param format: sample format, defaults to None as backward compatibility
            check the doc in :py:meth:`tensorcircuit.quantum.measurement_results`
            Six formats of measurement counts results:

                "sample_int": # np.array([0, 0])

                "sample_bin": # [np.array([1, 0]), np.array([1, 0])]

                "count_vector": # np.array([2, 0, 0, 0])

                "count_tuple": # (np.array([0]), np.array([2]))

                "count_dict_bin": # {"00": 2, "01": 0, "10": 0, "11": 0}
                    for cases d\in [11, 36], use 0-9A-Z digits (e.g., 'A' -> 10, ..., 'Z' -> 35);

                "count_dict_int": # {0: 2, 1: 0, 2: 0, 3: 0}

        :type format: Optional[str]
        :param random_generator: random generator,  defaults to None
        :type random_generator: Optional[Any], optional
        :param status: external randomness given by tensor uniformly from [0, 1],
            if set, can overwrite random_generator, shape [batch] for `allow_state=True`
            and shape [batch, nqubits] for `allow_state=False` using perfect sampling implementation
        :type status: Optional[Tensor]
        :param jittable: when converting to count, whether keep the full size. if false, may be conflict
            external jit, if true, may fail for large scale system with actual limited count results
        :type jittable: bool, defaults true
        :return: List (if batch) of tuple (binary configuration tensor and corresponding probability)
            if the format is None, and consistent with format when given
        :rtype: Any
        """
        # TODO(@refraction-ray): to be check:
        # 1. efficiency in different use case
        # 2. randomness interaction with jit when no explicit status
        if not allow_state:
            if random_generator is None:
                random_generator = backend.get_random_state()

            if batch is None:
                seed = backend.stateful_randu(random_generator, shape=[self._nqubits])
                r = self.perfect_sampling(seed)
                if format is None:  # batch=None, format=None, backward compatibility
                    return r
                r = [r]  # type: ignore
            else:
                r = []  # type: ignore
                if status is not None:
                    assert backend.shape_tuple(status)[0] == batch
                    for seed in status:
                        r.append(self.perfect_sampling(seed))  # type: ignore

                else:

                    @backend.jit
                    def perfect_sampling(key: Any) -> Any:
                        backend.set_random_state(key)
                        return self.perfect_sampling()

                    subkey = random_generator
                    for _ in range(batch):
                        key, subkey = backend.random_split(subkey)
                        r.append(perfect_sampling(key))  # type: ignore

            if format is None:
                return r
            r = backend.stack([ri[0] for ri in r])  # type: ignore
            ch = backend.cast(r, "int32")
            # ch = sample_bin2int(r, self._nqubits, dim=self._d)
        else:  # allow_state
            if batch is None:
                nbatch = 1
            else:
                nbatch = batch
            p = self.probability()

            # readout error
            if readout_error is not None:
                p = self.readouterror_bs(readout_error, p)
            ch = backend.probability_sample(nbatch, p, status, random_generator)
            # if random_generator is None:
            #     ch = backend.implicit_randc(a=a_range, shape=[nbatch], p=p)
            # else:
            #     ch = backend.stateful_randc(
            #         random_generator, a=a_range, shape=[nbatch], p=p
            #     )
            # confg = backend.mod(
            #     backend.right_shift(
            #         ch[..., None], backend.reverse(backend.arange(self._nqubits))
            #     ),
            #     2,
            # )
            if format is None:  # for backward compatibility
                confg = sample_int2bin(ch, self._nqubits, dim=self._d)
                prob = backend.gather1d(p, ch)
                r = list(zip(confg, prob))  # type: ignore
                if batch is None:
                    r = r[0]  # type: ignore
                return r
        if self._nqubits > 35:
            jittable = False
        return sample2all(
            sample=ch, n=self._nqubits, format=format, jittable=jittable, dim=self._d
        )

    def sample_expectation_ps(
        self,
        x: Optional[Sequence[int]] = None,
        y: Optional[Sequence[int]] = None,
        z: Optional[Sequence[int]] = None,
        shots: Optional[int] = None,
        random_generator: Optional[Any] = None,
        status: Optional[Tensor] = None,
        readout_error: Optional[Sequence[Any]] = None,
        noise_conf: Optional[Any] = None,
        nmc: int = 1000,
        statusc: Optional[Tensor] = None,
        **kws: Any,
    ) -> Tensor:
        """
        Compute the expectation with given Pauli string with measurement shots numbers

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.H(0)
        >>> c.rx(1, theta=np.pi/2)
        >>> c.sample_expectation_ps(x=[0], y=[1])
        -0.99999976
        >>> readout_error = []
        >>> readout_error.append([0.9,0.75])
        >>> readout_error.append([0.4,0.7])
        >>> c.sample_expectation_ps(x=[0], y=[1],readout_error = readout_error)

        >>> c = tc.Circuit(2)
        >>> c.cnot(0, 1)
        >>> c.rx(0, theta=0.4)
        >>> c.rx(1, theta=0.8)
        >>> c.h(0)
        >>> c.h(1)
        >>> error1 = tc.channels.generaldepolarizingchannel(0.1, 1)
        >>> error2 = tc.channels.generaldepolarizingchannel(0.06, 2)
        >>> readout_error = [[0.9, 0.75],[0.4, 0.7]]
        >>> noise_conf = NoiseConf()
        >>> noise_conf.add_noise("rx", error1)
        >>> noise_conf.add_noise("cnot", [error2], [[0, 1]])
        >>> noise_conf.add_noise("readout", readout_error)
        >>> c.sample_expectation_ps(x=[0], noise_conf=noise_conf, nmc=10000)
        0.44766843

        :param x: index for Pauli X, defaults to None
        :type x: Optional[Sequence[int]], optional
        :param y: index for Pauli Y, defaults to None
        :type y: Optional[Sequence[int]], optional
        :param z: index for Pauli Z, defaults to None
        :type z: Optional[Sequence[int]], optional
        :param shots: number of measurement shots, defaults to None, indicating analytical result
        :type shots: Optional[int], optional
        :param random_generator: random_generator, defaults to None
        :type random_generator: Optional[Any]
        :param status: external randomness given by tensor uniformly from [0, 1],
            if set, can overwrite random_generator
        :type status: Optional[Tensor]
        :param readout_error: readout_error, defaults to None. Overrided if noise_conf is provided.
        :type readout_error: Optional[Sequence[Any]]. Tensor, List, Tuple
        :param noise_conf: Noise Configuration, defaults to None
        :type noise_conf: Optional[NoiseConf], optional
        :param nmc: repetition time for Monte Carlo sampling for noisfy calculation, defaults to 1000
        :type nmc: int, optional
        :param statusc: external randomness given by tensor uniformly from [0, 1], defaults to None,
            used for noisfy circuit sampling
        :type statusc: Optional[Tensor], optional
        :return: [description]
        :rtype: Tensor
        """
        from .noisemodel import sample_expectation_ps_noisfy

        if noise_conf is None:
            inputs_nodes, _ = self._copy_state_tensor()
            inputs = inputs_nodes[0].tensor
            if self.is_dm is False:
                c = type(self)(self._nqubits, inputs=inputs)  # type: ignore
            else:
                c = type(self)(self._nqubits, dminputs=inputs)  # type: ignore
            if x is None:
                x = []
            if y is None:
                y = []
            if z is None:
                z = []
            for i in x:
                c.H(i)  # type: ignore
            for i in y:
                c.rx(i, theta=np.pi / 2)  # type: ignore
            s = c.state()  # type: ignore
            if self.is_dm is False:
                p = backend.abs(s) ** 2
            else:
                p = backend.abs(backend.diagonal(s))

            # readout error
            if readout_error is not None:
                p = self.readouterror_bs(readout_error, p)

            x = list(x)
            y = list(y)
            z = list(z)
            if shots is None:
                mc = measurement_counts(
                    p,
                    counts=shots,
                    format="count_vector",
                    random_generator=random_generator,
                    status=status,
                    jittable=True,
                    is_prob=True,
                )
                r = correlation_from_counts(x + y + z, mc)
            else:
                mc = measurement_counts(
                    p,
                    counts=shots,
                    format="sample_bin",
                    random_generator=random_generator,
                    status=status,
                    jittable=True,
                    is_prob=True,
                )
                r = correlation_from_samples(x + y + z, mc, self._nqubits)
            # TODO(@refraction-ray): analytical standard deviation
            return r
        else:
            return sample_expectation_ps_noisfy(
                c=self,
                x=x,
                y=y,
                z=z,
                noise_conf=noise_conf,
                nmc=nmc,
                shots=shots,
                statusc=statusc,
                status=status,
                **kws,
            )

    sexpps = sample_expectation_ps

    def readouterror_bs(
        self, readout_error: Optional[Sequence[Any]] = None, p: Optional[Any] = None
    ) -> Tensor:
        """Apply readout error to original probabilities of bit string and return the noisy probabilities.

        :Example:

        >>> readout_error = []
        >>> readout_error.append([0.9,0.75])  # readout error for qubit 0, [p0|0,p1|1]
        >>> readout_error.append([0.4,0.7])   # readout error for qubit 1, [p0|0,p1|1]


        :param readout_error: list of readout error for each qubits.
        :type readout_error: Optional[Sequence[Any]]. Tensor, List, Tuple
        :param p: probabilities of bit string
        :type p: Optional[Any]
        :rtype: Tensor
        """
        # if isinstance(readout_error, tuple):
        #     readout_error = list[readout_error]  # type: ignore
        try:
            nqubit = int(readout_error.shape[0])  # type: ignore
        except AttributeError:
            nqubit = len(readout_error)  # type: ignore
        readoutlist = []
        for i in range(nqubit):
            readoutlist.append(
                [
                    [readout_error[i][0], 1 - readout_error[i][1]],  # type: ignore
                    [1 - readout_error[i][0], readout_error[i][1]],  # type: ignore
                ]
            )
        readoutlist = backend.cast(
            backend.convert_to_tensor(readoutlist), dtype=dtypestr
        )

        ms = [Gate(readoutlist[i]) for i in range(nqubit)]
        p = backend.cast(p, dtypestr)
        p = Gate(backend.reshape2(p))
        for i in range(nqubit):
            ms[i][1] ^ p[i]
        nodes = ms + [p]
        r = contractor(nodes, output_edge_order=[m[0] for m in ms]).tensor
        p = backend.reshape(r, [-1])

        return backend.real(p)

    def replace_inputs(self, inputs: Tensor) -> None:
        """
        Replace the input state with the circuit structure unchanged.

        :param inputs: Input wavefunction.
        :type inputs: Tensor
        """
        inputs = backend.reshape(inputs, [-1])
        N = inputs.shape[0]
        n = _infer_num_sites(N, self._d)
        assert n == self._nqubits
        inputs = backend.reshape(inputs, [self._d] * n)
        if self.inputs is not None:
            self._nodes[0].tensor = inputs
            if self.is_dm:
                self._nodes[1].tensor = backend.conj(inputs)
        else:  # TODO(@refraction-ray) replace several start as inputs
            raise NotImplementedError("not support replace with no inputs")

    def cond_measurement(self, index: int, status: Optional[float] = None) -> Tensor:
        """
        Measurement on z basis at ``index`` qubit based on quantum amplitude
        (not post-selection). The highlight is that this method can return the
        measured result as a int Tensor and thus maintained a jittable pipeline.

        :Example:

        >>> c = tc.Circuit(2)
        >>> c.H(0)
        >>> r = c.cond_measurement(0)
        >>> c.conditional_gate(r, [tc.gates.i(), tc.gates.x()], 1)
        >>> c.expectation([tc.gates.z(), [0]]), c.expectation([tc.gates.z(), [1]])
        # two possible outputs: (1, 1) or (-1, -1)

        .. note::

            In terms of ``DMCircuit``, this method returns nothing and the density
            matrix after this method is kept in mixed state without knowing the
            measuremet resuslts



        :param index: the site index for the Z-basis measurement
        :type index: int
        :return: 0 or 1 for Z-basis measurement outcome
        :rtype: Tensor
        """
        return self.general_kraus(  # type: ignore
            [np.array([[1.0, 0], [0, 0]]), np.array([[0, 0], [0, 1]])],
            index,
            status=status,
            name="measure",
        )

    cond_measure = cond_measurement

    def to_graphviz(
        self,
        graph: graphviz.Graph = None,
        include_all_names: bool = False,
        engine: str = "neato",
    ) -> graphviz.Graph:
        """
        Not an ideal visualization for quantum circuit, but reserve here as a general approach to show the tensornetwork
        [Deprecated, use ``Circuit.vis_tex`` or ``Circuit.draw`` instead]
        """
        # Modified from tensornetwork codebase
        nodes = self._nodes
        if graph is None:
            # pylint: disable=no-member
            graph = graphviz.Graph("G", engine=engine)
        for node in nodes:
            if not node.name.startswith("__") or include_all_names:
                label = node.name
            else:
                label = ""
            graph.node(str(id(node)), label=label)
        seen_edges = set()
        for node in nodes:
            for i, edge in enumerate(node.edges):
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                if not edge.name.startswith("__") or include_all_names:
                    edge_label = edge.name + ": " + str(edge.dimension)
                else:
                    edge_label = ""
                if edge.is_dangling():
                    # We need to create an invisible node for the dangling edge
                    # to connect to.
                    graph.node(
                        "{}_{}".format(str(id(node)), i),
                        label="",
                        _attributes={"style": "invis"},
                    )
                    graph.edge(
                        "{}_{}".format(str(id(node)), i),
                        str(id(node)),
                        label=edge_label,
                    )
                else:
                    graph.edge(
                        str(id(edge.node1)),
                        str(id(edge.node2)),
                        label=edge_label,
                    )
        return graph

    def get_quvector(self) -> QuVector:
        """
        Get the representation of the output state in the form of ``QuVector``
        while maintaining the circuit uncomputed

        :return: ``QuVector`` representation of the output state from the circuit
        :rtype: QuVector
        """
        _, edges = self._copy()
        return QuVector(edges)

    quvector = get_quvector

    def projected_subsystem(self, traceout: Tensor, left: Tuple[int, ...]) -> Tensor:
        """
        remaining wavefunction or density matrix on sites in ``left``, with other sites
        fixed to given digits (0..d-1) as indicated by ``traceout``

        :param traceout: can be jitted
        :type traceout: Tensor
        :param left: cannot be jitted
        :type left: Tuple
        :return: _description_
        :rtype: Tensor
        """

        traceout = backend.convert_to_tensor(traceout, dtype=dtypestr)
        all_endns = onehot_d_tensor(traceout, d=self._d)
        nodes, front = self._copy()
        L = self._nqubits
        edges = []
        for i in range(len(traceout)):
            if i not in left:
                n = Gate(all_endns[i])
                nodes.append(n)
                front[i] ^ n[0]
            else:
                edges.append(front[i])

        if self.is_dm:
            for i in range(len(traceout)):
                if i not in left:
                    n = Gate(all_endns[i])
                    nodes.append(n)
                    front[i + L] ^ n[0]
                else:
                    edges.append(front[i + L])

        t = contractor(nodes, output_edge_order=edges)
        if self.is_dm:
            rho = backend.reshapem(t.tensor)
            return rho / (backend.trace(rho) + 1e-10)
        return backend.reshape(t.tensor / backend.norm(t.tensor), [-1])
