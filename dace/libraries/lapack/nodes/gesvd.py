# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import copy
import dace
import dace.library
import dace.properties
import dace.sdfg.nodes
from dace import dtypes
from dace.transformation.transformation import ExpandTransformation
from .. import environments
from dace.libraries.blas import environments as blas_environments
from dace.libraries.blas import blas_helpers


@dace.library.expansion
class ExpandGesvdPure(ExpandTransformation):
    """
    Naive backend-agnostic expansion of LAPACK GESVD.
    """

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        raise (NotImplementedError)


@dace.library.expansion
class ExpandGesvdOpenBLAS(ExpandTransformation):

    environments = [blas_environments.openblas.OpenBLAS]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_a, stride_a, rows_a,
         cols_a), desc_s, desc_u, desc_vt, desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_a.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == 'c':
            cast = "(lapack_complex_float*)"
        elif lapack_dtype == 'z':
            cast = "(lapack_complex_double*)"
        if desc_a.dtype.veclen > 1:
            raise (NotImplementedError)

        jobu = "'F'" if node.full_matrices else "'S'"
        jobvt = "'F'" if node.full_matrices else "'S'"

        m = rows_a
        n = cols_a
        min_mn = min(m, n)
        lda = stride_a
        ldu = m if node.full_matrices else m
        ldvt = n if node.full_matrices else min_mn

        code = f"_res = LAPACKE_{lapack_dtype}gesdd(LAPACK_ROW_MAJOR, {jobu}, {jobvt}, {m}, {n}, {cast}_xin, {lda}, _s, _u, {ldu}, _vt, {ldvt});"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandGesvdMKL(ExpandTransformation):

    environments = [blas_environments.intel_mkl.IntelMKL]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_a, stride_a, rows_a,
         cols_a), desc_s, desc_u, desc_vt, desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_a.dtype.base_type
        lapack_dtype = blas_helpers.to_blastype(dtype.type).lower()
        cast = ""
        if lapack_dtype == 'c':
            cast = "(MKL_Complex8*)"
        elif lapack_dtype == 'z':
            cast = "(MKL_Complex16*)"
        if desc_a.dtype.veclen > 1:
            raise (NotImplementedError)

        jobu = "'F'" if node.full_matrices else "'S'"
        jobvt = "'F'" if node.full_matrices else "'S'"

        m = rows_a
        n = cols_a
        min_mn = min(m, n)
        lda = stride_a
        ldu = m if node.full_matrices else m
        ldvt = n if node.full_matrices else min_mn

        code = f"_res = LAPACKE_{lapack_dtype}gesdd(LAPACK_ROW_MAJOR, {jobu}, {jobvt}, {m}, {n}, {cast}_xin, {lda}, _s, _u, {ldu}, _vt, {ldvt});"
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        return tasklet


@dace.library.expansion
class ExpandGesvdCuSolverDn(ExpandTransformation):

    environments = [environments.cusolverdn.cuSolverDn]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, n=None, **kwargs):
        (desc_a, stride_a, rows_a,
         cols_a), desc_s, desc_u, desc_vt, desc_result = node.validate(parent_sdfg, parent_state)
        dtype = desc_a.dtype.base_type
        veclen = desc_a.dtype.veclen

        func, cuda_type, _ = blas_helpers.cublas_type_metadata(dtype)
        func = func + 'gesvd'

        m = rows_a
        n = cols_a
        min_mn = min(m, n)

        jobu = "CUSOLVER_SIDE_LEFT_ALL" if node.full_matrices else "CUSOLVER_SIDE_LEFT"
        jobvt = "CUSOLVER_SIDE_RIGHT_ALL" if node.full_matrices else "CUSOLVER_SIDE_RIGHT"

        if veclen != 1:
            raise (NotImplementedError)

        code = (environments.cusolverdn.cuSolverDn.handle_setup_code(node) + f"""
                int __dace_workspace_size = 0;
                {cuda_type}* __dace_workspace;
                cusolverDn{func}BufferSize(
                    __dace_cusolverDn_handle, {jobu}, {jobvt}, {m}, {n}, {cuda_type}*,
                    {stride_a}, {cuda_type}*,
                    {cuda_type}*,
                    {cuda_type}*,
                    {cuda_type}*,
                    &__dace_workspace_size);
                cudaMalloc<{cuda_type}>(
                    &__dace_workspace,
                    sizeof({cuda_type}) * __dace_workspace_size);
                cusolverDn{func}(
                    __dace_cusolverDn_handle, {jobu}, {jobvt}, {m}, {n}, ({cuda_type}*)_xin,
                    {stride_a}, _s, _u,
                    {m}, _vt,
                    {min_mn}, __dace_workspace, __dace_workspace_size, _res);
                cudaFree(__dace_workspace);
                """)

        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        conn = tasklet.out_connectors
        conn = {c: (dtypes.pointer(dace.int32) if c == '_res' else t) for c, t in conn.items()}
        tasklet.out_connectors = conn

        return tasklet


@dace.library.node
class Gesvd(dace.sdfg.nodes.LibraryNode):

    implementations = {"OpenBLAS": ExpandGesvdOpenBLAS, "MKL": ExpandGesvdMKL, "cuSolverDn": ExpandGesvdCuSolverDn}
    default_implementation = None

    full_matrices = dace.properties.Property(dtype=bool, default=False)

    def __init__(self, name, full_matrices=False, *args, **kwargs):
        super().__init__(name, *args, inputs={"_xin"}, outputs={"_s", "_u", "_vt", "_res"}, **kwargs)
        self.full_matrices = full_matrices

    def validate(self, sdfg, state):
        in_edges = state.in_edges(self)
        if len(in_edges) != 1:
            raise ValueError("Expected exactly one input to gesvd")
        in_memlets = [in_edges[0].data]

        out_edges = state.out_edges(self)
        if len(out_edges) != 4:
            raise ValueError("Expected exactly four outputs from gesvd")

        squeezed1 = copy.deepcopy(in_memlets[0].subset)
        sqdims1 = squeezed1.squeeze()

        desc_a, desc_s, desc_u, desc_vt, desc_res = None, None, None, None, None
        for e in state.in_edges(self):
            if e.dst_conn == "_xin":
                desc_a = sdfg.arrays[e.data.data]
        for e in state.out_edges(self):
            if e.src_conn == "_s":
                desc_s = sdfg.arrays[e.data.data]
            if e.src_conn == "_u":
                desc_u = sdfg.arrays[e.data.data]
            if e.src_conn == "_vt":
                desc_vt = sdfg.arrays[e.data.data]
            if e.src_conn == "_result":
                desc_res = sdfg.arrays[e.data.data]

        if desc_a.dtype.base_type != desc_s.dtype.base_type:
            raise ValueError("Basetype of input and singular values must be equal!")
        if desc_a.dtype.base_type != desc_u.dtype.base_type:
            raise ValueError("Basetype of input and U must be equal!")
        if desc_a.dtype.base_type != desc_vt.dtype.base_type:
            raise ValueError("Basetype of input and VT must be equal!")

        stride_a = desc_a.strides[sqdims1[0]]
        shape_a = squeezed1.size()
        m = shape_a[0]
        n = shape_a[1]
        min_mn = min(m, n)

        if len(squeezed1.size()) != 2:
            raise ValueError("gesvd only supported on 2-dimensional arrays")

        return (desc_a, stride_a, m, n), desc_s, desc_u, desc_vt, desc_res
