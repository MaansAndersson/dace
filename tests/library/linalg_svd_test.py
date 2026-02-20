# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from dace import Memlet
from dace.libraries.linalg import Svd
import numpy as np
import pytest


def generate_matrix(m, n, dtype):
    from numpy.random import default_rng
    rng = default_rng(42)
    A = rng.random((m, n), dtype=dtype)
    return A


def make_sdfg(implementation, dtype, m, n, full_matrices=False, storage=dace.StorageType.Default):
    min_mn = min(m, n)

    if full_matrices:
        u_shape = [m, m]
        vt_shape = [n, n]
    else:
        u_shape = [m, min_mn]
        vt_shape = [min_mn, n]

    sdfg = dace.SDFG(f"linalg_svd_{implementation}_{dtype}_{m}x{n}")
    state = sdfg.add_state("dataflow")

    inp = sdfg.add_array("xin", [m, n], dtype)
    s_out = sdfg.add_array("sout", [min_mn], dtype)
    u_out = sdfg.add_array("uout", u_shape, dtype)
    vt_out = sdfg.add_array("vtout", vt_shape, dtype)

    xin = state.add_read("xin")
    sout = state.add_write("sout")
    uout = state.add_write("uout")
    vtout = state.add_write("vtout")

    svd_node = Svd("svd", full_matrices=full_matrices)
    svd_node.implementation = implementation

    state.add_memlet_path(xin, svd_node, dst_conn="_ain", memlet=Memlet.simple("xin", f"0:{m}, 0:{n}"))
    state.add_memlet_path(svd_node, sout, src_conn="_sout", memlet=Memlet.simple("sout", f"0:{min_mn}"))
    state.add_memlet_path(svd_node,
                          uout,
                          src_conn="_uout",
                          memlet=Memlet.simple("uout", f"0:{u_shape[0]}, 0:{u_shape[1]}"))
    state.add_memlet_path(svd_node,
                          vtout,
                          src_conn="_vtout",
                          memlet=Memlet.simple("vtout", f"0:{vt_shape[0]}, 0:{vt_shape[1]}"))

    return sdfg


@pytest.mark.parametrize("implementation, dtype, m, n, full_matrices, storage", [
    pytest.param("MKL", dace.float32, 4, 3, False, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, 4, 3, False, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float32, 3, 4, False, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, 3, 4, False, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("MKL", dace.float64, 4, 3, True, dace.StorageType.Default, marks=pytest.mark.mkl),
    pytest.param("OpenBLAS", dace.float32, 4, 3, False, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, 4, 3, False, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float32, 3, 4, False, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, 3, 4, False, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("cuSolverDn", dace.float32, 4, 3, False, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
    pytest.param("cuSolverDn", dace.float64, 4, 3, False, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
    pytest.param("cuSolverDn", dace.float32, 3, 4, False, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
    pytest.param("cuSolverDn", dace.float64, 3, 4, False, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
])
def test_svd(implementation, dtype, m, n, full_matrices, storage):
    sdfg = make_sdfg(implementation, dtype, m, n, full_matrices, storage)
    if implementation == 'cuSolverDn':
        sdfg.apply_gpu_transformations()
        sdfg.simplify()
    np_dtype = getattr(np, dtype.to_string())
    svd_sdfg = sdfg.compile()

    A = generate_matrix(m, n, np_dtype)
    min_mn = min(m, n)

    if full_matrices:
        u_shape = [m, m]
        vt_shape = [n, n]
    else:
        u_shape = [m, min_mn]
        vt_shape = [min_mn, n]

    S = np.zeros(min_mn, dtype=np_dtype)
    U = np.zeros(u_shape, dtype=np_dtype)
    VT = np.zeros(vt_shape, dtype=np_dtype)

    svd_sdfg(xin=A, sout=S, uout=U, vtout=VT, m=m, n=n)

    if dtype == dace.float32:
        rtol = 1e-5
    elif dtype == dace.float64:
        rtol = 1e-10
    else:
        raise NotImplementedError

    A_reconstructed = U @ np.diag(S) @ VT
    error = np.linalg.norm(A - A_reconstructed) / np.linalg.norm(A)
    assert error < rtol, f"SVD reconstruction error {error} exceeds tolerance {rtol}"


###############################################################################

if __name__ == "__main__":
    test_svd("MKL", dace.float32, 4, 3, False, dace.StorageType.Default)
    test_svd("MKL", dace.float64, 4, 3, False, dace.StorageType.Default)
    test_svd("MKL", dace.float64, 4, 3, True, dace.StorageType.Default)
    test_svd("cuSolverDn", dace.float32, 4, 3, False, dace.StorageType.GPU_Global)
    test_svd("cuSolverDn", dace.float64, 4, 3, False, dace.StorageType.GPU_Global)
