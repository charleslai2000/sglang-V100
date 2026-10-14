Status: DONE
Work area: research
Objective: 完成G022/T002的SM70 GPTQ Marlin donor复用性、预打包数据映射、数值语义、资源预算与下一阶段建议定向调查；不改生产实现或启动新实验。
Inputs: G022 goal/plan及已提交E026/E028证据；main当前源文件；`scripts/setup_v100_marlin.sh`与Marlin V100兼容补丁；固定donor revision `6d72a49939701d26b15b617a4cd2423174adb2d1`。
Completion: 提交可复核报告，明确当前S1/G017/SM70 Dense接通状态、实际热路径、预处理边界、语义与资源预算、唯一下一阶段方案及验收标准；精确提交并推送到当前main，核对远端SHA。
Result: 报告见`control/G022-falcon-v100-c4-performance/T002-SM70-GPTQ-PREPACK-REPORT.md`。裁决NO-GO-DONOR：已审核的marlin_v100 Dense接口是implicit symmetric uint4b8，不能承载实际GPTQ-v1逐输出qzeros与stored-zero+1；SGLang当前SM70 Dense helper只用于NVFP4/FP8路径，GPTQLinearKernel没有连至它。可复用的是候选的压缩Q4/FP32 accumulator+split-K思路、activation gather与静态元数据计划，不是donor GEMM/repack实现。唯一建议：下阶段以G017精确SHA的GPTQ-v1 Q4/g128候选为基线，完成T002剩余同负载GPU-forward/service/cache/4K ledger闭环，不做Marlin Dense移植或生产默认变更。
Checks: 核验当前V100仓库main原HEAD `dca488908ee4e3f1bc676c3bf5dcd26ff049cfc3`；仅快进同步已审阅的origin/main三个README文档提交后，调查HEAD为`2d06a902aeba2d9cf38d80f1a916107d0520a2f5`。工作树干净，V100 repository无mainline G022目录，相关任务与E026/E028资料从main历史commit读取。MemAvailable约61.1GiB；主机可见RTX3090空闲，不在授权目标V100上，不使用任何GPU。未加载checkpoint tensors；报告逐项标明数据缺口。未执行安装脚本/build/test/service/profile。
Remaining: 无（本调查交付完成；目标阶段测量与实现仍按G022现有ACTIVE frontier进行）。