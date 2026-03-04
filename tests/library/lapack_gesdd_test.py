# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.libraries.lapack as lapack
import dace.libraries.standard as std
from dace.sdfg import sdfg
import numpy as np
import pytest

from dace.memlet import Memlet


###############################################################################


def make_sdfg(implementation, dtype, storage=dace.StorageType.Default):

    n = dace.symbol("n")
    m = dace.symbol("m")

    suffix = "_device" if storage != dace.StorageType.Default else ""
    transient = storage != dace.StorageType.Default

    sdfg = dace.SDFG("matrix_svdfa_gesdd_{}_{}".format(implementation, str(dtype)))
    state = sdfg.add_state("dataflow")

    xhost_arr = sdfg.add_array("x", [n, m], dtype, storage=dace.StorageType.Default)
    uhost_arr = sdfg.add_array("u", [n, n], dtype, storage=dace.StorageType.Default)
    shost_arr = sdfg.add_array("s", [n], dtype, storage=dace.StorageType.Default)
    vthost_arr = sdfg.add_array("vt", [n, n], dtype, storage=dace.StorageType.Default)

    if transient:
        x_arr = sdfg.add_array("x" + suffix, [n, m], dtype, storage=storage, transient=transient)
        xt_arr = sdfg.add_array('xt' + suffix, [m, n], dtype, storage=storage, transient=transient)

        u_arr = sdfg.add_array('u' + suffix, [m, m], dtype, storage=storage, transient=transient)
        ut_arr = sdfg.add_array('ut' + suffix, [m, m], dtype, storage=storage, transient=transient)
       
        vt_arr = sdfg.add_array('vt' + suffix, [n, n], dtype, storage=storage, transient=transient)
        vtt_arr = sdfg.add_array('vtt' + suffix, [n, n], dtype, storage=storage, transient=transient)

        s_arr = sdfg.add_array("s" + suffix, [n], dtype, storage, transient=transient )

    
    sdfg.add_array("result" + suffix, [1], dace.dtypes.int32, storage=storage, transient=transient)
    result = state.add_access("result" + suffix)
 

    if transient:
        xhi = state.add_read("x")
        #xho = state.add_write("x")
        xi = state.add_access("x" + suffix)
        #xo = state.add_access("x" + suffix)
        xin = state.add_access("xt" + suffix)
        
        uout = state.add_access("ut" + suffix)
        sout = state.add_access("s" + suffix)
        vtout = state.add_access("vtt" + suffix)
        
        uh = state.add_access("u")   
        u = state.add_access("u" + suffix)             
        sh = state.add_access("s")   
        vth = state.add_access("vt")
        vt = state.add_access("vt" + suffix)

        transpose_in = std.Transpose("transpose_in", dtype=dtype)
        transpose_in.implementation = "cuBLAS"

        # U and VT are output in column major, so we need to transpose them back to row major
        transpose_out_u = std.Transpose("transpose_out_u", dtype=dtype)
        transpose_out_u.implementation = "cuBLAS"
        transpose_out_vt = std.Transpose("transpose_out_vt", dtype=dtype)
        transpose_out_vt.implementation = "cuBLAS"

        state.add_nedge(xhi, xi, Memlet.from_array(*xhost_arr))
        state.add_memlet_path(xi, transpose_in, dst_conn='_inp', memlet=Memlet.from_array(*x_arr))
        state.add_memlet_path(transpose_in, xin, src_conn='_out', memlet=Memlet.from_array(*xt_arr))
        
        state.add_memlet_path(uout, transpose_out_u, dst_conn='_inp', memlet=Memlet.from_array(*ut_arr))
        state.add_memlet_path(transpose_out_u, u, src_conn='_out', memlet=Memlet.from_array(*u_arr)) 
        state.add_nedge(u, uh, Memlet.from_array(*uhost_arr))
        
        state.add_memlet_path(vtout, transpose_out_vt, dst_conn='_inp', memlet=Memlet.from_array(*vtt_arr))
        state.add_memlet_path(transpose_out_vt, vt, src_conn='_out', memlet=Memlet.from_array(*vt_arr)) 
        state.add_nedge(vt, vth, Memlet.from_array(*vthost_arr))

        state.add_nedge(sout, sh, Memlet.from_array(*shost_arr))
    else:
        xin = state.add_access("x" + suffix)
        uout = state.add_access("u" + suffix)
        sout = state.add_access("s" + suffix)
        vtout = state.add_access("vt" + suffix)

    gesvd_node = lapack.Gesvd("gesvd", full_matrices=True)
    gesvd_node.implementation = implementation

    # in
    state.add_memlet_path(xin, gesvd_node, dst_conn="_xin", memlet=Memlet.simple(xin, "0:n, 0:n", num_accesses=n * n))
    # out
    state.add_memlet_path(gesvd_node, result, src_conn="_res", memlet=Memlet.simple(result, "0", num_accesses=1))
    state.add_memlet_path(gesvd_node, uout, src_conn="_u", memlet=Memlet.simple(uout,"0:m, 0:m", num_accesses=m*m))
    state.add_memlet_path(gesvd_node, sout, src_conn="_s", memlet=Memlet.simple(sout,"0:n", num_accesses=n))
    state.add_memlet_path(gesvd_node, vtout, src_conn="_vt", memlet=Memlet.simple(vtout,"0:n, 0:n", num_accesses=n*n))

    return sdfg

@pytest.mark.parametrize("implementation, dtype, storage", [
    pytest.param("OpenBLAS", dace.float32, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("OpenBLAS", dace.float64, dace.StorageType.Default, marks=pytest.mark.lapack),
    pytest.param("cuSolverDn", dace.float32, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
    pytest.param("cuSolverDn", dace.float64, dace.StorageType.GPU_Global, marks=pytest.mark.gpu),
    ])

def test_gesdd(implementation, dtype, storage):
    sdfg = make_sdfg(implementation, dtype, storage)
    gesdd_sdfg = sdfg.compile()
    np_dtype = getattr(np, dtype.to_string())

    from scipy.linalg import svd
    A = np.array([[16,2,3,13], [5,11,10,8], [9, 7,6,12], [4,14, 15,  4]], dtype=np_dtype)
    U, s, Vh = svd(A)

    #print(U)
    u = np.zeros((4,4), dtype=np_dtype)
    s = np.array([0,0,0,0], dtype=np_dtype)
    vt = A.copy() #np.zeros((4,4),dtype=dtype)
    
    lapack_status = np.array([-1], dtype=np.int32)
    gesdd_sdfg(x=A, result=lapack_status, u=u, vt=vt, s=s, n=4, m=4)

    print(U-u)


#test_gesdd("OpenBLAS", dace.float64, dace.StorageType.Default)
#test_gesdd("OpenBLAS", dace.float32, dace.StorageType.Default)
test_gesdd("cuSolverDn", dace.float64, dace.StorageType.GPU_Global)
test_gesdd("cuSolverDn", dace.float32, dace.StorageType.GPU_Global)
