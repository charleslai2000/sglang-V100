#!/usr/bin/env python3
"""CPU-only, source-level numerical audit of the uploaded L43 down GPTQ fixture.

No GPU, PyTorch, CUDA extension, benchmark or production modification is involved.
Requires Python 3, NumPy, and a C++ compiler with IEEE float FMA support.
Run: python audit.py --fixture /path/to/packed-word-fixture-package --out ./results
"""
from __future__ import annotations
import argparse
import csv
import ctypes
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np

SOURCE_URL = 'https://raw.githubusercontent.com/charleslai2000/sglang-V100/44882a0a3acde81d270cac2980b8af3ad8a6eac8/sgl-kernel/csrc/gemm/gptq/gptq_candidate_v8.cu'

def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def pointer(array: np.ndarray):
    if not array.flags.c_contiguous:
        raise ValueError('C array must be contiguous')
    return array.ctypes.data_as(ctypes.c_void_p)

def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def checked_tensor(root: Path, name: str, spec: dict) -> np.ndarray:
    path = root / f'{name}.bin'
    if sha_file(path) != spec['sha256']:
        raise ValueError(f'{name}: original manifest SHA mismatch')
    dtype = {'torch.float16':'<f2','torch.float32':'<f4','torch.int32':'<i4'}[spec['dtype']]
    a = np.fromfile(path, dtype=dtype)
    shape = tuple(spec['shape'])
    if a.size != int(np.prod(shape)):
        raise ValueError(f'{name}: size mismatch')
    return a.reshape(shape)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.fixture.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'manifest.json').read_text())
    arrays = {k:checked_tensor(root,k,v) for k,v in manifest['tensors'].items()}
    checked_tensor(root,'s1_output',manifest['s1_output'])
    if arrays['x'].shape != (4,12288) or arrays['scales'].shape != (96,3072):
        raise ValueError('This audit is intentionally scoped to uploaded L43/down M4')
    x0 = arrays['x'].astype(np.float32)
    M,K = x0.shape
    G,N = arrays['scales'].shape
    q = arrays['qweight'].view(np.uint32)
    z = arrays['qzeros'].view(np.uint32)
    scales = arrays['scales'].astype(np.float32)
    g = arrays['g_idx']
    if not (np.all((g>=0)&(g<G)) and K%128==0):
        raise ValueError('Invalid GPTQ shape/group index')
    xstress = np.fromfile(root/'c57_stress_x_fp16.bin',dtype='<f2').reshape(M,K)
    if not same_bits(xstress,(x0*4).astype(np.float16)):
        raise ValueError('Saved stress input differs from (x*4).half()')
    saved = np.fromfile(root/'c57_stress_fp32_output.bin',dtype='<f4').reshape(M,N)
    candidate = np.fromfile(root/'packed_candidate_stress_fp32_output.bin',dtype='<f4').reshape(M,N)
    saved_ref = np.fromfile(root/'stress_fp32_reference.bin',dtype='<f4').reshape(M,N)
    for arm in ('c57','packed_candidate'):
        if (root/f'{arm}_stress_x_fp16.bin').read_bytes() != (root/'c57_stress_x_fp16.bin').read_bytes():
            raise ValueError('Per-arm stress input differs')
        if (root/f'{arm}_stress_fp32_reference.bin').read_bytes() != (root/'stress_fp32_reference.bin').read_bytes():
            raise ValueError('Per-arm reference differs')
    if not same_bits(saved,candidate):
        raise ValueError('Saved arm outputs are not byte-identical')
    w = np.empty((K,N),np.float32)
    columns = np.arange(N)
    w_exact = True
    for k in range(K):
        code = ((q[k//8] >> np.uint32((k%8)*4)) & 15).astype(np.int32)
        zero = ((z[g[k],columns//8] >> ((columns%8)*4)) & 15).astype(np.int32)+1
        w[k] = (code-zero).astype(np.float32)*scales[g[k]]
        w_exact &= np.array_equal(w[k].astype(np.float64),(code-zero).astype(np.float64)*scales[g[k]].astype(np.float64))
    if not w_exact:
        raise ValueError('FP32 dequantization is not exact for this fixture')
    source = Path(__file__).with_name('reproduce.cpp')
    libpath = out/'reproduce_cpu.so'
    command = ['g++','-O3','-march=native','-fPIC','-shared','-fno-fast-math','-ffp-contract=off',str(source),'-o',str(libpath)]
    subprocess.run(command,check=True)
    lib = ctypes.CDLL(str(libpath))
    lib.partials.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*4
    for name in ('reduce32','reduce64'):
        getattr(lib,name).argtypes = [ctypes.c_void_p]*2+[ctypes.c_int]*2
    lib.reduce_kahan.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2
    lib.partial_split4.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*4
    lib.partial_dot2.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*3
    if lib.rounding_mode() != 0:
        raise RuntimeError('Expected round-to-nearest on the audit host')
    P=K//128
    def reduce(part: np.ndarray, method: str) -> np.ndarray:
        y = np.empty((M,N),np.float64 if method=='reduce64' else np.float32)
        if method=='kahan':
            c = np.empty_like(y)
            lib.reduce_kahan(pointer(part),pointer(y),pointer(c),P,M*N)
        else:
            getattr(lib,method)(pointer(part),pointer(y),P,M*N)
        return y.astype(np.float32)
    def score(y: np.ndarray, ref: np.ndarray) -> dict:
        allowed = np.float32(1e-4)+np.float32(5e-5)*np.abs(ref)
        error = np.abs(y-ref)
        bad = (~np.isfinite(y)) | (error>allowed)
        return {'strict_bad':int(bad.sum()),'max_error_over_tolerance':float(np.max(error/allowed)),
                'bad_points':np.argwhere(bad).tolist(),'output_sha256':sha_array(y),
                'finite_fp32':bool(np.isfinite(y).all()),'finite_final_fp16':bool(np.isfinite(y.astype(np.float16)).all())}
    result = {'scope':'CPU-only; L43/down M4, original and 4x stress; no GPU implementation/timing/Graph/service claim',
              'source_reference':SOURCE_URL,'source_file_sha256':sha_file(source),
              'audit_script_sha256':sha_file(Path(__file__)),
              'build_command':command,'cpu_binary_sha256':sha_file(libpath),
              'environment':{'python':platform.python_version(),'numpy':np.__version__,'machine':platform.machine()},
              'input_files':{p.name:sha_file(p) for p in sorted(root.iterdir()) if p.is_file()},
              'checks':{'all_original_manifest_hashes_match':True,'saved_arm_outputs_byte_identical':True,
                        'per_arm_input_and_reference_identical':True,'fp32_dequantization_exact':True,
                        'nonidentity_gidx':not np.array_equal(g,np.arange(K)//128),
                        'group_counts':np.bincount(g).tolist()},'runs':{}}
    kept = {}
    for factor in (1,4):
        x = (x0*factor).astype(np.float16).astype(np.float32)
        p32 = np.empty((P,M,N),np.float32);p64 = np.empty((P,M,N),np.float64)
        lib.partials(pointer(x),pointer(w),pointer(p32),pointer(p64),M,K,N,128)
        ref = p64.sum(axis=0).astype(np.float32)
        original = reduce(p32,'reduce32')
        if factor==4:
            if not same_bits(original,saved) or not same_bits(ref,saved_ref):
                raise RuntimeError('Source reproduction / independent oracle does not match saved outputs')
        variants = {'original_fp32_fma_reduce32':original,
                    'original_partial_reduce64':reduce(p32,'reduce64'),
                    'original_partial_kahan':reduce(p32,'kahan')}
        accurate_partial = p64.astype(np.float32)
        for method in ('reduce32','reduce64','kahan'):
            variants['fp64_partial_rounded_fp32_'+method]=reduce(accurate_partial,method)
        dot = np.empty_like(p32)
        lib.partial_dot2(pointer(x),pointer(w),pointer(dot),M,K,N)
        dot_identical = same_bits(dot,accurate_partial)
        for method in ('reduce32','reduce64','kahan'):
            variants['dot2_fp32_partial_'+method]=reduce(dot,method)
        for mode,name in ((0,'contiguous'),(1,'interleaved')):
            sp = np.empty_like(p32)
            lib.partial_split4(pointer(x),pointer(w),pointer(sp),M,K,N,mode)
            for method in ('reduce32','reduce64','kahan'):
                variants['four_short_fma_'+name+'_'+method]=reduce(sp,method)
        checks={name:score(y,ref) for name,y in variants.items()}
        result['runs'][str(factor)]={'input_sha256':sha_array(x.astype(np.float16)),
                                  'reference_sha256':sha_array(ref),'outputs_tested_per_variant':M*N,
                                  'dot2_partial_matches_fp64_rounded':dot_identical,'variants':checks}
        kept[factor]=(original,ref)
        if factor==4:
            exact = p64.sum(axis=0)
            rounded_partial_sum = p32.astype(np.float64).sum(axis=0)
            rows=[]
            for m,n in np.argwhere(np.abs(original-ref)>(np.float32(1e-4)+np.float32(5e-5)*np.abs(ref))):
                m,n=int(m),int(n)
                # Exact dyadic products: finite half values are multiples of 2^-24.
                xin=(x[m].astype(np.float64)*2**24).astype(np.int64)
                wi=(w[:,n].astype(np.float64)*2**24).astype(np.int64)
                terms=[int(u)*int(v) for u,v in zip(xin,wi)]
                num=sum(terms);ev=num/(2**48)
                ep=np.array([sum(terms[j:j+128])/(2**48) for j in range(0,K,128)])
                assert np.float32(ev)==ref[m,n]
                assert np.array_equal(ep,p64[:,m,n])
                row={'m':m,'n':n,'actual_fp32':float(original[m,n]),'reference_fp32':float(ref[m,n]),
                     'exact_numerator':str(num),'exact_denominator':str(2**48),
                     'exact_value':ev,'sum_abs_products':sum(map(abs,terms))/(2**48),
                     'cancellation_ratio':sum(map(abs,terms))/abs(num),
                     'signed_partial_accumulation_error':float(rounded_partial_sum[m,n]-exact[m,n]),
                     'signed_final_reduction_error':float(original[m,n]-rounded_partial_sum[m,n]),
                     'signed_reference_rounding_term':float(exact[m,n]-ref[m,n]),
                     'signed_total_error':float(original[m,n])-float(ref[m,n]),
                     'allowed_fp32':float(np.float32(1e-4)+np.float32(5e-5)*np.abs(ref[m,n]))}
                rows.append(row)
            result['eight_point_decomposition']=rows
            with (out/'eight_point_decomposition.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            for name,y in variants.items():
                if name in ('original_fp32_fma_reduce32','fp64_partial_rounded_fp32_reduce32','dot2_fp32_partial_kahan'):
                    y.astype('<f4').tofile(out/(name+'.bin'))
    result['checks']['original_4x_output_is_exactly_4_times_1x']=same_bits(kept[4][0],kept[1][0]*4)
    result['checks']['reference_4x_output_is_exactly_4_times_1x']=same_bits(kept[4][1],kept[1][1]*4)
    (out/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
    for factor,run in result['runs'].items():
        print('factor',factor)
        for name,v in run['variants'].items():
            print(name,v['strict_bad'],f"{v['max_error_over_tolerance']:.9f}")
    print('Results:',out/'audit_results.json')
if __name__=='__main__':
    main()
