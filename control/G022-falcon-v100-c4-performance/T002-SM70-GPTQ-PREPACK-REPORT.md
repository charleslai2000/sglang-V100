# G022/T002 — SM70 GPTQ 预打包路线定向调查

## 裁决：NO-GO-DONOR

不建议把`zhinianqin/marlin_v100` Dense直接接入当前 Falcon GPTQ线性热路径。决定性不兼容是：现有SGLang的SM70 Dense GEMM按Marlin symmetric `uint4b8`类型解释Q4，并把零点留在隐式中心；Falcon真实GPTQ checkpoint仍需逐group/逐output channel读取`qzeros`，并按GPTQ-v1语义使用`stored_zero + 1`。已审核的donor Dense op没有证据表明接受并消费SGLang `GPTQMarlinLinearKernel`当前传入的zero-point tensor/显式g_idx；直接把数据重排成现有donor格式会丢失非固定zero-point信息。不得因`is_sym=true`臆定所有实际zero均为8。Donor的SM70 GEMM/repack不宜直接复用；可复用候选的紧凑Q4读取、加载期layout变换、按g_idx gather和FP32 split-K累加思想，但应沿当前GPTQ-v1数学合同建立明确的新layout/路径。

此裁决只针对**当前公开Dense接口按原样复用**，不证明所有可能重构都不可能，也不是性能认证。G017 accepted Candidate的历史证据继续有效，S1默认与G017精确SHA opt-in保持不变。此次仅提交调查文档，无源码、kernel、dispatcher、配置/default改动，无新服务/性能实验。

## 1. 当前主线、源码身份与完整调用链

**调查对象与证据更新**：首次源码审计基线为`2d06a902aeba2d9cf38d80f1a916107d0520a2f5`；随后personal/main已推进至`621824f82e3d9ba29c7725135d5ae61ff02e706c`，包含E034、G017 exact-final SHA与真实checkpoint静态审计证据。本报告本次收敛以该personal/main历史材料为准。`origin/main`仍旧且对当前凭据返回403；未强推。其他V100工作树中的无关dirty状态须保留。V100上的S1/G017是另一套既有运行产物/路径身份，不可用main源码HEAD或旧`02f923ea...`实验二进制冒充accepted final c57。

**S1**：Falcon GPTQ被选为普通GPTQ后，由`GPTQConfig.get_quant_method`→`GPTQLinearMethod`→`GPTQLinearScheme`→`GPTQLinearKernel`。权重加载到packed qweight、qzeros、scales、g_idx；`process_weights_after_loading`调用`gptq_shuffle`（是否有效取决于`use_shuffle`/desc_act）；实际Forward每次调用`gptq_gemm(x,qweight,qzeros,scales,g_idx,use_shuffle,bits)`。这是基础GPTQ CUDA路径，不是Marlin。

**显式/可转换GPTQ-Marlin**：`GPTQMarlinConfig.override_quantization_method`只对`user_quant is None | marlin | gptq_marlin`且配置可转换的checkpoint选择Marlin；用户显式`quantization=gptq`保持GPTQ。Marlin scheme加载期由`GPTQMarlinLinearKernel.process_weights_after_loading`排序g_idx、在`g_idx_sort_indices`保存原K置换、repack qweight、permute scales、zero_points并分配workspace；apply经`apply_gptq_marlin_linear`传activation、packed weights、scale、zero、g_idx及排序索引。该调用最终是`gptq_marlin_gemm`或compile-context wrapper，**不是**`_sm70_marlin_v100_gemm_op`。

**SM70 donor Dense状态**：`scripts/setup_v100_marlin.sh`固定默认`MARLIN_V100_REF=6d72a49939701d26b15b617a4cd2423174adb2d1`，构建并安装MoE与可选Dense `.so`；脚本不是运行时GEMM dispatch。`marlin_utils.py`的`_sm70_marlin_v100_available`、`_sm70_marlin_v100_repack_ops`、`_sm70_marlin_v100_gemm_op`提供探测、共享库装载和op查找。检查实际引用点可见helper的dense GEMM调用用于`marlin_utils_fp8.py`，不是GPTQ-Marlin线性apply。`gptq_kernels.py`会借helper覆盖`gptq_marlin_repack`，但GPTQ Marlin linear的apply仍调用in-tree `gptq_marlin_gemm`；SGLang源码明确stock Marlin的repack/GEMM在SM70是空stub，故“donor repack可用”不等同“Donor Dense GPTQ完整dispatch已接通”。现有可见SM70 donor路径主要服务MoE WNA16、FP8/NVFP4等范围。

**G017**：已接受split-K GPTQ Candidate的代码提交`5fac74206b...`可在历史中查看，后来由`5c5dda7a29`整合。其分支代码将opt-in候选加在基础`gptq_gemm`小M路由中，默认保持S1；它不是Marlin Dense，也没有启用`setup_v100_marlin.sh`的Dense kernel。G017 final extension SHA `02f923ea7329912a4413f179f292acb444a5e001caebf110b24f531cabf70d8b`、strict pre-cast FP32 oracle/ID/service证据详见G017计划和`experiments/E026-local-v100-s4-m3c/results.md`历史材料。它能供G022复用为比较基线，但本次没有重新生成或验证该binary。

> 当前main不含G022控制树和E026/E028工作文件；它们存在于main可达历史提交。本报告及T002 task是本次加入main的首个G022/T002本地控制记录。

## S1/G017 A/B 与现有 CUPTI Forward 矩阵

**版本与验收边界**：S1 extension SHA `fea05241f2d588bf5a54f03a898360a49fbe0f256e25f59abbd9fd10b9257842`；accepted G017 Candidate SHA `c57e50dc58c59773f701b8a4260140c47ba18a94db8e1b497b6496ab4a84acb9`（显式 opt-in，S1 default未动）。旧G017 build-close candidate `02f923ea...`不作为当前accepted binary身份。E026/E028证明accepted候选的六个真实L0/L22/L43 gate_up/down fixture cast前FP32 accumulator对FP64 GPTQ-v1 oracle 0 bad，阈值`abs(err)<=1e-4+5e-5*abs(ref)`未改；18个1K/4K×c1/c2/c4×3波次的256-token greedy ID均与S1匹配。FP16 production output是该accumulator的half cast，不声称其直接通过cast前oracle；高幅stress至8× finite，16×的L43/down overflow。

| 现存S1 vs accepted G017 Candidate | S1 | G017 Candidate | 观测结论/边界 |
|---|---:|---:|---|
| 1K progressive GPU Event c1 mean | 28.537ms | 26.360ms | -2.177ms / -7.63% |
| 1K progressive GPU Event c2 mean | 31.430ms | 29.067ms | -2.363ms / -7.52% |
| 1K progressive GPU Event c4 mean | 34.124ms | 31.719ms | -2.405ms / -7.05% |
| 1K aggregate service output TPS c1/c2/c4 | 33.70 / 59.38 / 103.87 | 36.38 / 63.71 / 110.88 tok/s | +7.95 / +7.21 / +6.73% |
| 4K aggregate service output TPS c1/c2/c4 | 29.67 / 46.64 / 47.73 | 31.78 / 49.34 / 50.74 tok/s | +7.10 / +5.70 / +6.27%；c4服务并不证明GPU实际batch=4 |
| Matched GPTQ NCU M1/M2/M4 | gate_up 252.38/262.82/290.14µs；down 161.86/172.45/185.98µs | gate_up 216.22/219.14/230.82µs；down 144.35/142.34/147.36µs | 仅named GPTQ kernels；long-scoreboard下降，DRAM writes/math/not-selected升高；非Forward aggregate |

GPU Event文件身份与runtime SHA在E028记录中核过；Event集是1K progressive KV长度1025–4352，不能扩写成4K GPU step收益，且FPM没有request-ID到client response join。服务TPS、TTFT/ITL与CUDA Event不是同一计量边界。E026最终服务JSON在报告中记录hash；其accepted Candidate route/3波次结果是已验收历史结果，E028另存2波次对应JSON摘要，不能把不同样本数拼成新的统一统计。Prefill/Cold-Warm cache、每cell KV/Mamba/VRAM容量收益未有完整matched S1-vs-Candidate账本。

| S1 CUPTI 12.8 ApplicationReplay Forward（每格三次均值） | Event ms | SM Active % | SM throughput %peak | Tensor active %peak | DRAM effective read+write GB/s | DRAM throughput %peak | L2 sector hit % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefill EXTEND 1K bs1 | 285.22 | 92.278 | 41.853 | 41.278 | 230.1 | 25.742 | 86.837 |
| Prefill EXTEND 4K bs1 | 1119.78 | 97.065 | 45.200 | 44.786 | 212.1 | 23.736 | 85.389 |
| Decode Graph1 bs1 | 30.36 | 74.353 | 25.365 | 0 | 269.4 | 29.879 | 66.980 |
| Decode Graph2 bs2 | 33.35 | 75.690 | 25.759 | 2.112 | 254.1 | 28.096 | 58.156 |
| Decode Graph4 bs4 | 36.21 | 77.422 | 26.798 | 1.905 | 256.3 | 27.892 | 60.178 |
| Decode Graph4 bs4 Candidate | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

矩阵来源`control/experiments/G022/cupti-forward-utilization.md`（T001已接受PASS，七指标向量每项3次、L2 throughput每项另2次15-pass）。范围为固定fixture的单Forward annotation；DRAM GB/s是raw byte/Event的range平均并由独立DRAM-throughput公式交叉校验，不是时间序列峰值。Candidate侧没有同方法/同load的CUPTI向量；不从operator NCU拼凑SM/Tensor/DRAM全Forward A/B。L2是设备counter，不是HiCache hit/cold-cache恢复指标。

**明确的现存缺口**：accepted c57 exact final SHA的4K per-step CUDA Event A/B；FPM-to-request/response ID join；actual c4 GPU batch=4的Candidate Forward；Candidate同负载CUPTI全Forward SM/Tensor/DRAM/L2对照；Prefill TPS及Cold/Warm cache TTFT/HiCache恢复对照；逐测点VRAM、KV/Mamba容量和完整exclusive owner ledger。Kernel-duration/trace sums可重叠，不能直接扣为独占节省。G017现有1K GPU Event和1K/4K aggregate service增益足以维持其既有accepted opt-in，但不构成新Kernel/预打包路线的收益理由；按本次授权不新开GPU任务，也不开发实现。

## 2. 每轮工作与可前移预处理

真实热路径不是每轮完整解包/重排权重：qweight/qzeros/scales/g_idx在device侧常驻；当前`gptq_gemm`及G017候选在每个Forward都读取打包Q4，并做必要的group/zero/scale索引、activation布局/gather和乘累加。G017候选把若干工作改成与M相关的activation gather和split-K partial/reduction；其性能和数学边界已有历史测量，但不是Donor。

加载期已做：参数读取/TP shard、packed tensor装载、weight参数登记；Marlin分支会再做排序、qweight repack、scale permute、zero-point变换及workspace分配。基础S1具体是否改变qweight，必须由其kernel配置`use_shuffle`语义与实测loader状态共同判定，不能只按通用注释称“每轮shuffle”。E028 CPU checkpoint audit核实了真实存储映射，但没有运行时对象/内存追踪来证明S1每轮进行何种重排；故**不声称当前Falcon权重确实在每轮被全量重排**。

若为新压缩Q4 kernel预打包，加载期候选映射：
- 不展开FP16矩阵；原始逻辑`Q[K,N]`保留每元素4-bit，物理每字节两个code。构造目标tile顺序的紧凑Q4 bytes，layout约`[K_tile,N_tile,packed-nibble]`或能让连续lane联合取相邻N值的等价布局；pad尺寸/尾区只加code padding，不生成全精度权重。
- 按真实GPTQ `g_idx[k]`建立稳定`perm[k]`使group连续，预排qweight K轴为`Q[perm[k],n]`。同时保存inverse permutation或明确其不需要；forward激活必须做`x_perm[k]=x[perm[k]]`，保证内积仍是`sum_k x[perm[k]]*Q[perm[k],n]`。如果不重排activation，则不能单独重排权重K轴。对于desc_act=false且group本已连续，identity映射应避免多余g_idx gather。
- qzeros的逻辑维度是group×N，先按GPTQ-v1 unpack `stored_zero+1`获得真实zero，再把zero和scales按**相同N tile**排列；scale/zero仍按原group映射，不因对称标志压成常数。若K reorder跨组，packed code、g_idx、scale group与zero group必须一致地置换或转成连续group-id查表。
- Falcon fused gate_up是concatenated N维输出。先按原始checkpoint列界分别解码gate和up；布局打包可在物理上共享N tile，但不能让一侧的scale/zero索引或column offset跨越另一侧错误复用。要求映射等式逐列成立，且split位置/TP partition与真实layer loader完全一致。

**Checkpoint核验状态由personal/main中的E028 CPU审计闭合，不标UNKNOWN：**对象为Falcon-H1-7B-Instruct-GPTQ-Int4，revision `52036ae497a2c0c253e17f98cc8652f2f7082ae3`；全模型132个线性模块，GPTQ-v1 Q4/group128、`sym=true`、`desc_act=true`、`static_groups=false`。132个模块共38,928,384个存储zero nibble全为7，GPTQ-v1有效zero均为8；811,008个g_idx中794,243项不符合简单`K//128`顺序。K=3072矩阵有24组，K=12288矩阵有96组，scale为有限FP16。Gate/up在已审layers 0/22/43中g_idx/qzeros相等，但scales不等，gate/up必须独立保留scale映射。准确来源及hash见`experiments/E028-g022-evidence-driven-optimization/sm70-gptq-prepack-investigation.md`（SHA256 `426a25e797c36a808ad22ea19a4eb85544fb5ed2f009637dd64914c037480e25`）。观测zero值不授权把一般实现硬编码为8：通用语义仍为stored-zero+1。Donor checkout/binary本次未运行。

## 3. 数值合同与具体风险

冻结数学语义是GPTQ-v1 Q4 group128：`q=(packed_qweight >> nibble_shift)&15`，zero按GPTQ-v1解包后`zero=stored_zero+1`，scale由真实group与输出列定位，dequant值`(q-zero)*scale`；activation按同一个K置换映射；Oracle对FP64 reference逐元素要求`abs(err)<=1e-4+5e-5*abs(reference)`，不允许修改。参考路径必须保留**cast前FP32 accumulator**，FP16输出与FP32 oracle满足该极严绝对/相对界在数学上可能不可能，不能拿FP16接近当PASS。G017历史oracle结果只支持其精确候选和指定FP32 accumulator边界，不自动证明Marlin donor。

Donor不兼容不只是“证据未充分”：personal/main中的`sm70-gptq-prepack-investigation.md`对固定revision的源码审计记录Dense `marlin_gemm`显式拒绝非空`g_idx_or_none`/`perm_or_none`（报错SM70不支持act_order），而Falcon真实checkpoint为desc_act=true、static_groups=false，且多数g_idx非K顺序。donor Q4 `uint4`零点消费接口要求FP16 zeros；SGLang传入的Marlin zero metadata为packed integer形式，V100 dense wrapper也未证明会转换成donor需要的FP16-zero contract。即便以某种方式绕过loader guard，Dense GEMM的activation ordering和zero-point ABI仍不匹配。Donor还只暴露FP16 output，且其split-K epilogue路径是FP16 atomic accumulation；这不提供冻结合同要求的可观测cast前FP32 accumulator。故对**当前Falcon真实配置及接口不变**，NO-GO由具体接口/数据序机制冲突支持，不是单纯“尚未测试”。对可能重写act-order、qzero变换与FP32 accumulator的新消费者不作普遍不可能判断。

**最小离线判别用例**（未来若开发兼容消费者时才需；本阶段未执行）：逐元素校验原始Q4/qzero+1/scale/g_idx到目标格式的整数及metadata映射，含真实act-order、实际非8 zero若存在、融合gate/up列界；随后针对FP64-derived FP32 reference运行cast前累加器冻结阈值。当前接口已不满足act-order/qzero ABI门，无须启动该判别实验。

## 4. 显存/资源与M1/M2/M4成本

没有真实Falcon模型tensor shape/checkpoint bytes在本目录可读；所以不给虚构逐层MiB合计。精确预算公式如下，统计时按各实际量化矩阵`K_i,N_i,G_i=ceil(K_i/128)`求和，TP shard优先使用本地partition shape：

- 原始Q4：`ceil(K_i*N_i/2)` bytes；qzeros 4-bit存储约`ceil(G_i*N_i/2)` bytes（具体checkpoint packing/block padding须读取tensor shape）；scales约`G_i*N_i*scale_dtype_bytes`；g_idx若int32约`4*K_i`。以上是张量payload静态预算，不含对象/allocator对齐。
- 紧凑目标Q4预打包：有效code仍约`ceil(K_i*N_i/2)` bytes，tile/N/K pad开销按`Kpad*Npad/(K*N)-1`乘payload；scale+zero映射元数据应尽量复用或以group/N tile压缩，`perm`/inverse perm若各用int32则至多`4*K_i` bytes/份。若同时保留原始及packed权重，静态驻留峰值至少是两份Q4加各自元数据；允许在成功检查点后释放旧Q4/可重建临时副本，不能默认双份永久驻留。
- FP32 partial workspace：对M行、N列、S个K-split，若partial为FP32，则`4*M*N*S` bytes；若另有reduction/output临时区，另计`4*M*N`或按实际dtype计。不得把此run workspace误算为每层常驻权重。实际G017 split数/确切每geometry临时tensor应按已接受源码和manifest重新计算。
- 工作集峰值需分列：checkpoint/load CPU RAM；load期原始+packed双格式及转换临时峰值；转换完成释放旧格式后的推理常驻权重；每Forward activation gather和split-K partial/reduction；CUDA graph capture保存的输出/临时buffer；fallback原权重是否保留。无FP16完整weight materialization。

M成本（静态路径分析，非新测）：M1每CTA可复用activation一次但模型持续存在固定launch/metadata开销；M2/M4增加CTA内activation reuse机会，也相应增大partial/临时结果与reduce工作。`g_idx`非连续时activation gather的读访问可能不合并；若预打包已稳定排序，可降低动态索引但仍需加载/构造`x_perm`，该收益须计入额外launch或融合gather成本。K/N tile padding导致inactive lane/CTA工作；M不足tile时M=1/2/4可能存在输出行或SM CTA分布浪费。拆分K改善CTA并行度却增加partial写流量、workspace与reduction成本。T002历史M1/M2/M4 NCU表明G017 GPTQ candidate在对应fixture减少long-scoreboard、增加DRAM写和math/not-selected pressure，但那些是kernel scope，不提供Donor的CTA浪费或全Forward收益。

已有证据只允许有边界地预算：E028记录G017候选在1K GPU Event c1/c2/c4分别减少2.177/2.363/2.405ms，服务TPS增益和精确SHA在该归档中；c4 GPTQ gate_up/down owner duration sums约18.7ms属于可能重叠的trace分类，不是可直接扣除的独占收益；E028同样记录4K GPU-step A/B、cache/Prefill净收益及request-ID FPM/service join缺失。不得把这些写成新Marlin收益；本阶段未做性能实验。

## 5. 唯一建议与下一阶段验收

**唯一建议：NO-GO-DONOR，不移植`marlin_v100` Dense GPTQ GEMM；复用G017候选的紧凑Q4/FP32 split-K技术方向，下一阶段只完成T002剩余exact-SHA比较账本与G022已授权的最小闭环，保持S1 default与Candidate opt-in。**

下一阶段最小范围：不改S1 dispatcher/default，不增加新Dense donor；按已存在G017精确SHA S1/Candidate复用T002的真实c1/c2/c4数据与E028/E026记录，只补足冻结验收明确缺少的4K per-step GPU Event对照、必要的实际Batch身份/同工作负载服务关联、Prefill及Cold/Warm cache影响、memory/capacity和owner budget未知项。是否需要新增运行应由实际缺口和资源门决定；不重复T001已PASS CUPTI向量或M1/M2/M4匹配NCU。

验收不变：真实Falcon checkpoint；S1/G017 extension SHA精确；Q4/g128/qzero+1/g_idx与融合映射通过静态整数code核对；cast前FP32 oracle逐元素遵守原阈值；graph capture/replay和greedy token/state正确性；各测点记录actual GPU batch、CUDA Event、service TPS/TTFT/ITL、Prefill/cache、VRAM/KV/Mamba capacity、rollback；S1 default不变。单算子更快或旧ACK均不能替代全路径证据。

## 路径、检查与交付身份

- 本报告：`control/G022-falcon-v100-c4-performance/T002-SM70-GPTQ-PREPACK-REPORT.md`
- 同目录task：`control/G022-falcon-v100-c4-performance/tasks/T002-sm70-gptq-prepack-investigation.md`
- 关键源：`python/sglang/srt/layers/quantization/gptq/gptq.py`、`.../gptq/schemes/gptq_linear.py`、`.../gptq/schemes/gptq_marlin.py`、`python/sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py`、`python/sglang/srt/layers/quantization/marlin_utils.py`、`.../marlin_utils_fp8.py`、`scripts/setup_v100_marlin.sh`、`patches/marlin-v100-sm70.patch`。
- Donor固定revision：`6d72a49939701d26b15b617a4cd2423174adb2d1`（仅核对安装脚本固定值；本机无checkout，无远端源码/扩展SHA可核）。
- 静态源码复核：GPTQ scheme→Kernel apply与Marlin helper引用路径；stock `gptq_marlin_repack.cuh`/`gptq_marlin.cuh`标记`__CUDA_ARCH__<800`分支。没有调用donor脚本、不build、不安装、不启动服务、不使用GPU。Checkpoint CPU tensor audit来自personal/main E028证据，复核了其摘要/映射结论；本次未重跑tensor extraction。没有运行donor binary或新误差/性能实验。V100 repo在任务开始前main HEAD `dca488908ee4e3f1bc676c3bf5dcd26ff049cfc3`；审阅`origin/main`三个提交均仅文档后快进，调查代码基线`2d06a902aeba2d9cf38d80f1a916107d0520a2f5`。报告/task首次提交`cfc51ce934e2f048322ce5da3a3faa152c9a75a9`，尚未推送：origin/main拒绝当前凭据(403)；personal/main在推送时前进且含同期G022变更，当前需精确整合报告到最新有权限main后再推，未强推。现状/剩余见同目录task文件。SGLang production/kernel源码无改动。
