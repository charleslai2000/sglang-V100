#!/usr/bin/env python3
"""Freeze already-observed E035 FP64 execution-chain provenance; performs no experiments."""
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[1]
patches=root/'patches'; snapshots=root/'patch-archive'/'source-snapshots';snapshots.mkdir(parents=True,exist_ok=True)
files={
 'python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py':'python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py',
 'sgl-kernel/python/sgl_kernel/load_utils.py':'sgl-kernel/python/sgl_kernel/load_utils.py',
 'sgl-kernel/CMakeLists.txt':'sgl-kernel/CMakeLists.txt',
 'sgl-kernel/csrc/common_extension.cc':'sgl-kernel/csrc/common_extension.cc',
 'experiments/E026-local-v100-s4-m3c/scripts/build_candidate_extension.sh':'experiments/E026-local-v100-s4-m3c/scripts/build_candidate_extension.sh',
}
def sha(b):return hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
base='8d2a01455f6e9804b85e906535839962cb2be812'
if git('rev-parse','HEAD') != base: raise SystemExit('Unexpected baseline; archive refuses to guess')
for rel in files:
 b=subprocess.check_output(['git','-C',str(repo),'show',f'{base}:{rel}'])
 p=snapshots/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
manifest={'archive_created_before_revert_commit':base,'baseline_commit':base,'baseline_commit_tree':git('rev-parse',f'{base}^{{tree}}'),'patches':{},'patched_source_snapshots':{},'runtime':{'model_revision':'52036ae497a2c0c253e17f98cc8652f2f7082ae3','model_config_sha256':'ca8c91f8a18a4917f9580ce1cbcb015a6f7cee2b24022cf6c3e8c94d6943a7f7','s1_common_ops_sha256':'fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842','candidate_extension_path':'/mnt/data/pc01/sglang-v100-migration/experiments/E035/numerical-fix-fp64-optin-build/g022_fp64_optin.so','candidate_extension_sha256':'35998ff295f0a4fe27417886ff2fd86bc9a1c1030975435e8b34aa42fb26b5f6','cuda':'12.8','pytorch':'2.9.1+cu128','gpu_uuid':'GPU-eaeb0cc6-58aa-8413-bef3-7fb075992189','source_commit_for_shared_patch':'8d2a01455f6e9804b85e906535839962cb2be812'},'service_order':'complete c57 matrix, then complete FP64 matrix; three repeats per cell; not alternating pairs','limits':['No complete hermetic environment lock or exact server binary/log hashes in repository.','Launcher and process-local runtime shims do not independently recreate all machine state.','Do not claim bit-for-bit experiment reproduction from this archive.','Shared integration patch was observed against baseline commit only for these experiment hunks; any simultaneous background edits not attributable from this diff are excluded.'],'artifacts':{'c57_service_json_sha256':'b67e1ddae9cb551dc11189eee039d399fce9b70e76efb5036624293ed2e203d7','fp64_service_json_sha256':'c000b623d7da32bdef0dc432d10f23812bdab4740dc4d4378bd69a8af63e1617','strict_matrix_sha256':'163c2e0a4bbe7d630e8942ba88df00f481ad6cc944d54ebcf6b454034ae38cb8','state_isolation_sha256':'3af1a6a85f25e2d16ff7611cb4ac3e6b279fefaf1fdde9e13fb56ea3467469a7'}}
for p in sorted(patches.glob('*.patch')):manifest['patches'][p.name]={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
for rel in files:
 src=repo/rel
 if src.exists(): b=src.read_bytes()
 else: b=subprocess.check_output(['git','-C',str(repo),'show',f'{base}:{rel}'])
 # archive exact observed modified form from patch application only when present in patch.
 patchname='fp64-E026-build-argument.patch' if 'build_candidate_extension' in rel else 'fp64-shared-integration.patch'
 if patchname.endswith('build-argument.patch'):
  b=subprocess.check_output(['git','-C',str(repo),'show',f'{base}:{rel}'])+b''
  patch=subprocess.check_output(['git','-C',str(repo),'show',f'{base}:{rel}'])
  # Recreate modified file by applying saved patch in memory is intentionally avoided.
 # Modified source hashes are recorded in README from pre-revert scan.
 sp=snapshots/rel
 manifest['patched_source_snapshots'][rel]={'baseline_sha256':sha(sp.read_bytes()),'note':'Modified form is represented by archived patch plus baseline snapshot; modified full-file snapshot unavailable.'}
(root/'patch-archive'/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('wrote patch-archive/manifest.json')
