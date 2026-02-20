# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
import numpy as np

from dace import Memlet
from dace.libraries.lapack import Gesvd
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.lapack import environments
from dace.libraries.blas import environments as blas_environments


def _make_sdfg(node, parent_state, parent_sdfg, implementation):
    arr_desc = node.validate(parent_sdfg, parent_state)
    (in_shape, in_dtype, in_strides, out_s_shape, out_u_shape, out_vt_shape, out_dtype, storage, m, n,
     min_mn) = arr_desc

    sdfg = dace.SDFG("{l}_sdfg".format(l=node.label))

    ain_arr = sdfg.add_array('_ain', in_shape, dtype=in_dtype, strides=in_strides, storage=storage)
    aout_arr = sdfg.add_array('_aout', in_shape, dtype=in_dtype, transient=True, storage=storage)
    s_arr = sdfg.add_array('_s', out_s_shape, dtype=out_dtype, transient=True, storage=storage)
    u_arr = sdfg.add_array('_u', out_u_shape, dtype=out_dtype, transient=True, storage=storage)
    vt_arr = sdfg.add_array('_vt', out_vt_shape, dtype=out_dtype, transient=True, storage=storage)
    info_arr = sdfg.add_array('_info', [1], dtype=dace.int32, transient=True, storage=storage)

    state = sdfg.add_state("{l}_state".format(l=node.label))

    gesvd_node = Gesvd('gesvd', full_matrices=node.full_matrices)
    gesvd_node.implementation = implementation

    ain = state.add_read('_ain')
    aout1 = state.add_read('_aout')
    aout2 = state.add_access('_aout')
    sout = state.add_write('_s')
    uout = state.add_write('_u')
    vtout = state.add_write('_vt')
    info = state.add_write('_info')

    state.add_nedge(ain, aout1, Memlet.from_array(*ain_arr))
    state.add_memlet_path(aout2, gesvd_node, dst_conn="_xin", memlet=Memlet.from_array(*aout_arr))
    state.add_memlet_path(gesvd_node, info, src_conn="_res", memlet=Memlet.from_array(*info_arr))
    state.add_memlet_path(gesvd_node, sout, src_conn="_s", memlet=Memlet.from_array(*s_arr))
    state.add_memlet_path(gesvd_node, uout, src_conn="_u", memlet=Memlet.from_array(*u_arr))
    state.add_memlet_path(gesvd_node, vtout, src_conn="_vt", memlet=Memlet.from_array(*vt_arr))

    return sdfg


@dace.library.expansion
class ExpandSvdPure(ExpandTransformation):

    environments = []

    @staticmethod
    def make_sdfg(node, parent_state, parent_sdfg):
        raise NotImplementedError("Missing pure implementation of linalg.svd.")

    @staticmethod
    def expansion(node, state, sdfg):
        node.validate(sdfg, state)
        if node.dtype is None:
            raise ValueError("Data type must be set to expand " + str(node) + ".")
        return ExpandSvdPure.make_sdfg(node, state, sdfg)


@dace.library.expansion
class ExpandSvdOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "OpenBLAS")


@dace.library.expansion
class ExpandSvdMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "MKL")


@dace.library.expansion
class ExpandSvdCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return _make_sdfg(node, parent_state, parent_sdfg, "cuSolverDn")


@dace.library.node
class Svd(dace.sdfg.nodes.LibraryNode):

    implementations = {"OpenBLAS": ExpandSvdOpenBLAS, "MKL": ExpandSvdMKL, "cuSolverDn": ExpandSvdCuSolverDn}
    default_implementation = None

    full_matrices = dace.properties.Property(dtype=bool, default=False)
    compute_uv = dace.properties.Property(dtype=bool, default=True)

    def __init__(self, name, full_matrices=False, compute_uv=True, *args, **kwargs):
        super().__init__(name, *args, inputs={"_ain"}, outputs={"_sout", "_uout", "_vtout"}, **kwargs)
        self.full_matrices = full_matrices
        self.compute_uv = compute_uv

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to svd")
        in_memlet = in_edges[0].data
        out_edges = state.out_edges(self)
        if len(out_edges) != 3:
            raise ValueError("Expected exactly three outputs from svd")

        squeezed_in = copy.deepcopy(in_memlet.subset)
        dims_in = squeezed_in.squeeze()

        desc_ain, desc_s, desc_u, desc_vt = None, None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_ain":
                desc_ain = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_sout":
                desc_s = sdfg.arrays[e.data.data]
            if e.src_conn == "_uout":
                desc_u = sdfg.arrays[e.data.data]
            if e.src_conn == "_vtout":
                desc_vt = sdfg.arrays[e.data.data]

        if desc_ain.dtype.base_type != desc_s.dtype.base_type:
            raise ValueError("Basetype of input and singular values must be equal!")
        if desc_ain.dtype.base_type != desc_u.dtype.base_type:
            raise ValueError("Basetype of input and U must be equal!")
        if desc_ain.dtype.base_type != desc_vt.dtype.base_type:
            raise ValueError("Basetype of input and VT must be equal!")

        if len(squeezed_in.size()) != 2:
            raise ValueError("linalg.svd only supported on 2-dimensional arrays")

        shape_in = squeezed_in.size()
        m, n = shape_in[0], shape_in[1]
        min_mn = min(m, n)

        strides_in = np.array(desc_ain.strides)[dims_in].tolist()
        if strides_in[-1] != 1:
            raise ValueError("Matrices with column strides greater than 1 are unsupported")

        if self.full_matrices:
            s_shape = [min_mn]
            u_shape = [m, m]
            vt_shape = [n, n]
        else:
            s_shape = [min_mn]
            u_shape = [m, min_mn]
            vt_shape = [min_mn, n]

        return (shape_in, desc_ain.dtype, strides_in, s_shape, u_shape, vt_shape, desc_s.dtype, desc_ain.storage, m, n,
                min_mn)
