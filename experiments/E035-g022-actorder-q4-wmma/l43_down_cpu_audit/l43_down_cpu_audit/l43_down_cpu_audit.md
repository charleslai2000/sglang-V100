# L43/down：packed-word 数值问题独立 CPU 复核

## 结论与范围

读取用户上传的原始张量，在 CPU 重新解包、反量化、逐次模拟 c57 的 128-K FP32 FMA 与 96 段顺序 FP32 归约。没有运行 CUDA、用户 GPU、Graph、Forward 或服务。

确认：保存的 c57 与 packed-word 候选 FP32 输出逐字节相同；12,288 个输出全部按原算法逐位复现，包含同样八个 strict failure。当前输入下是共有累加误差，不是 packed-word 改造新增的数值差异；没有整模型能力退化的测量结论。

本例的算术修正可行；GPU 成本、编译结果、跨夹具及整模型正确性尚未测量。原候选 NUMERICAL-NO-GO 不因此变成生产采纳。

## 1. 数据核验

原始 manifest 的 x/qweight/qzeros/scales/g_idx 及 s1_output SHA 全匹配。M=4、K=12288、N=3072；96 个 GPTQ group 各含128项，g_idx 非恒等。qzeros 全为 0x77777777，GPTQ-v1 stored-zero +1 得到有效 zero=8。

显式 stress `(x*4).half()` 与保存输入逐位相同，最大幅值9680；两臂输入和 oracle 副本分别逐字节相同。逐元素验证 `(q-zero)*FP16_scale` 在本夹具可由 FP32 精确表示，排除该步反量化舍入为误差来源。

| 对象 | SHA-256 |
|---|---|
| stress FP16 input | `100faa2bc448c9bbbd17e68206ade236ce65dd0394fc0a953639a1c55d6439f1` |
| 两臂保存的 FP32 output，以及 CPU 复现 output | `4a0800c14d8c49750c1582e99c546908ce9284c8ef01decf3dd23d7252a61ff8` |
| 保存的 FP32 oracle，以及 CPU 独立计算 oracle | `da7ee5fa596912cc9ec2ed61f4015b9b46427cd29acc3dd61fc42e49a0f47602` |

此包没有原始 GPU 执行器或 `.so`，不冒称再次核验服务器实际加载状态。以上是上传文件与独立 CPU 计算间的验证。

## 2. 复现与独立参考

按原 K 顺序每128项做 FP32 FMA，然后96个 partial 顺序 FP32 相加。split index 不等于 GPTQ g_idx；非恒等 g_idx 仍逐项选择 scale/zero。

CPU C++ 显式 `std::fma(float,float,float)`，round-to-nearest；编译 `-O3 -march=native -fno-fast-math -ffp-contract=off`，不允许普通加法重结合。独立 FP64 partial/GEMM→FP32 oracle 与包内 oracle 全部逐位一致。

八个失败点另外做任意精度整数求和：有限 FP16 输入及 scales 都是 $2^{-24}$ 的整数倍，含整数 `(q-zero)` 的乘积统一以 $2^{48}$ 为分母。八点 reference 均与精确整数结果一致；八点每个128-K partial 的 FP64结果也匹配整数求和。

原幅值 strict-bad=0，4×为8。实际 $y_4=4y_1$、$r_4=4r_1$ 均逐位成立，输入4×没有额外舍入或溢出。因此是误差放大后越过含固定绝对项的冻结阈值，不是溢出；不构成放宽容差的理由。

## 3. 误差分解

$$y-r=(\sum_jp_j-\sum_jp_j^*)+(y-\sum_jp_j)+(\sum_jp_j^*-r).$$

三项依次为128-K分段累加误差、96段最终归约误差、oracle的最终FP32舍入项。$p_j$ 为复现FP32 partial，$p_j^*$ 为高精度partial，$r$ 为冻结FP32 oracle。下面是有符号输出误差，不是耗时。

| 点 | 分段累加误差 | 最终归约误差 | 总误差 | 允许误差 |
|---|---:|---:|---:|---:|
| (0,80) | -0.000636780409 | -0.000111758709 | -0.000748634338 | 0.00073054462 |
| (0,1034) | +0.000291892888 | +3.9935112e-06 | +0.000295877457 | 0.000167904509 |
| (0,1442) | +0.000157474853 | +4.23043966e-05 | +0.000199794769 | 0.000143530429 |
| (0,2806) | -0.000363227904 | -0.000211954117 | -0.000575065613 | 0.000444227306 |
| (1,3069) | -0.000251546717 | -8.5234642e-05 | -0.000336766243 | 0.00017685465 |
| (2,1119) | -0.000216211418 | -0.000206276774 | -0.000422477722 | 0.000403274666 |
| (3,272) | -5.99708219e-05 | -0.000193268061 | -0.000253260136 | 0.000145115162 |
| (3,2962) | +0.000204446274 | +8.51750374e-05 | +0.000289678574 | 0.000201055518 |

(0,1034)主要来自分段累加，(3,272)主要来自最终归约，不能把所有反例归给 reducer。完整舍入项和精确分子见 CSV/JSON。

八点的抵消比 $\sum_k|x_kw_k|/|\sum_kx_kw_k|$ 为约1117–12685，仅描述这些点正负项抵消，不是模型质量指标。

## 4. 算术反事实：4×输入，12,288个输出

所有变体使用相同输入和GPTQ权重，不变的 `1e-4 + 5e-5*abs(reference)` 容差；未进行GPU参数扫表。

| 算术变体 | strict-bad | 最大误差/容差 |
|---|---:|---:|
| 原FP32 partial + 原FP32归约 | 8 | 1.904198 |
| 原partial，仅末级改FP64 | 6 | 1.738038 |
| 原partial，仅末级改FP32 Kahan | 5 | 1.540662 |
| FP64段内→FP32 partial，原归约 | 0 | 0.955249 |
| FP64段内→FP32 partial，FP64归约 | 0 | 0.461736 |
| FP64段内→FP32 partial，FP32 Kahan归约 | 0 | 0.265848 |
| FP32 Dot2段内，原归约 | 0 | 0.955249 |
| FP32 Dot2段内，FP32 Kahan归约 | 0 | 0.265848 |
| 四条连续32项FMA，原归约 | 3 | 1.567261 |
| 四条交错32项FMA，原归约 | 2 | 2.682262 |

全部变体也实算了原幅值，均strict-bad=0，完整结果见JSON。

只修末级归约仍失败；缩短FMA链也不是充分修复。提高128-K段内累加精度可清除本例反例；但沿用原reducer时最大误差/容差为0.955，余量较小。配合Kahan后为0.266。这些是观测值，不是一般误差上界。

FP32 Dot2 使用 FMA 保留乘积残差、TwoSum保留加法残差，不是仅给末级求和加补偿。其1,179,648个partial与FP64段内和舍入到FP32在本夹具1×/4×均逐位一致。

## 5. 后续方案依据与风险

后续应修既有SIMT的分段累加，而非重开已拒绝WMMA。有限可行路线为：128-K内FP64累加，或FP32 Dot2补偿，仍以FP32保存partial，并采用稳定末级归约。输入、Q4压缩格式、g_idx、zero/scales数学语义均不必改变。

这不是GPU方案已采纳。FP64可能增加寄存器、类型转换及算术成本；Dot2增加指令和寄存器，编译重结合还可能破坏补偿；Kahan也增加末级依赖。哪条在V100成本较低尚不确定，不能从CPU数值结果推出性能。

下一阶段应一次完成数值修复、GPU成本与受影响路径验证，按最终结果裁决；不得让新实现复制已证实超容差的c57误差来满足位型一致，但模型输出变化仍必须核验。

## 6. 复现

```bash
python audit.py --fixture /path/to/packed-word-fixture-package --out ./results
```

依赖NumPy及g++。脚本检查原始SHA，在结果目录编译CPU辅助库，不接触GPU或生产仓库。证据包包含脚本、C++、JSON、八点CSV、三个代表性FP32输出；不重复附带权重，不包含GPU二进制。

## 7. 公开方法依据

- 冻结源码 `gptq_candidate_v8.cu`，commit `44882a0a3acde81d270cac2980b8af3ad8a6eac8`，函数 `packed_fp32_partial` / `packed_fp32_reduce`：
  https://raw.githubusercontent.com/charleslai2000/sglang-V100/44882a0a3acde81d270cac2980b8af3ad8a6eac8/sgl-kernel/csrc/gemm/gptq/gptq_candidate_v8.cu
- NVIDIA CUDA 12.8, Floating Point and IEEE 754：FMA单次舍入、点积运算次序对精度的影响。
  https://docs.nvidia.com/cuda/archive/12.8.0/floating-point/index.html
- Ogita, Rump, Oishi (2005), Accurate Sum and Dot Product, SIAM J. Sci. Comput. 26(6):1955–1988；Algorithm 3.5 TwoProductFMA、5.3 Dot2。
  https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf

公开方法支持补偿算法，不提供该Falcon/V100路径的性能承诺。
