## Work Areas

* `control/` — manage Goals, Plans, Tasks, current frontier, and cross-session integration; do not perform substantial professional work here.
* `design/` — develop and maintain architecture, proposals, and confirmed design decisions.
* `review/` — independently review an explicitly assigned claim, deliverable, or stage; review is not a mandatory workflow step.
* `experiments/` — create new empirical evidence through controlled experiments, benchmarks, reproductions, or prototypes.
* `ops/` — manage runtime environments, services, deployment, infrastructure, operational procedures, and material operational records.
* `research/` — investigate and synthesize existing evidence and external sources; create an experiment when new measurements are required.

Ordinary implementation, debugging, and direct verification remain repo-root work unless a more specific directory instruction applies.


针对这几天反复出现的问题，设定以下硬约束：
1. 严禁只完成源码审查、编译检查、少量单测，就结束任务。
2. 严禁因为 fixture、IPC、Profiler、Launcher、LSP 或普通测试失败而停工等待新指令。
3. 严禁在 GPU 已获授权且资源可用时，未经真实运行验证就宣布本轮工作结束。
4. 严禁把 PARTIAL、NOT_RUN、NOT_VERIFIED 当作正常的阶段性交付。
5. 严禁反复认证已通过的 ACK、锁、缓存恢复和单节点驱逐，偏离当前性能目标。
6. 严禁用静态推测、跨 Run 拼接或不完整遥测冒充真实 GPU 性能数据。
7. 严禁汇报一次进度就停止执行；汇报后必须继续推进。
8. 严禁擅自建立新工作树、清理既有 Dirty Tree，或把未经验收的代码混入提交。
只有真正的架构/Ownership 决策冲突，或无法自主解决的外部硬阻断，才允许停工。

# 控制文档

控制文档必须精简，每个文档只说一个事项，内容必须简明扼要，严禁长篇大论，严禁在一个文件中反复添>加内容，文档名称应当体现内容，以方便寻找，markdown文档应当保持精细的内部标题目录结构，以方便查找。
