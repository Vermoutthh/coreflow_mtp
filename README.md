# CoreFlow MTP

与旧 `CoreFlow_` 并行的 CoreFlow 实现：保留 workflow DAG、context 分区、全局 GPU 分配及缓存管理，执行层使用 `coreflow_mtp` 环境中的 vLLM 0.18.1 V1 和真实模型权重。原仓库与已安装 vLLM 文件不修改。

MTP 使用本地 Qwen3.5-9B，K 在实例启动时固定；K=0 为普通自回归。改变 K 需要重新部署实例，不接受请求级临时切换。

## 仓库范围

本仓库只发布核心源码、正式运行／profiling／分配入口、设计文档和小型配置示例。不包含实验采集脚本、实验结果、数据集、模型权重、运行时缓存或私有配置。`results/` 和 `run/` 是本地运行目录，不随 Git 上传；`examples/` 中的 JSON 是手工配置，不是数据集或测量结果。

使用前需要自行准备匹配的推理环境、模型权重及 workflow 输入，并将示例中的模型绝对路径和下文的环境路径替换为实际位置。

## 目录

```text
coreflow_mtp/
├── src/coreflow_mtp/
│   ├── allocator/      # 请求容量、context 分区与全局分配
│   ├── predictor/      # 成本预测；带来源校验的 K=0 模型
│   ├── profiler/       # K=0 Llama 组件 profiling
│   ├── runtime/        # 真实推理、配置、监控与执行接口
│   ├── cache/          # 显式驻留、CPU 换出、同机 KV 迁移
│   ├── dag/            # workflow 依赖和真实输入输出
│   ├── controller.py
│   ├── scheduler.py
│   └── tools.py
├── scripts/            # 正式运行、profiling、分配入口
├── examples/           # allocation 与 profiling 配置示例
├── docs/               # 设计与缓存协议
├── results/            # 本地结果目录，不随 Git 上传
└── run/                # 本地实例状态与运行时缓存，不随 Git 上传
```

本地研究工作区另外保存 MTP 研究脚本、smoke／验收脚本和实验结果，它们不属于此代码仓库。下列 `scripts/` 保留的是系统本身的正式入口。

## 环境

```bash
cd /path/to/coreflow_mtp
```

下文使用 `/opt/conda/envs/coreflow_mtp/bin/python -I -B`。`-I` 隔离外部 Python 路径，各入口只添加本项目 `src`。不要导入旧 `CoreFlow_/scripts/common.py`，也不要升级旧环境。GPU 引擎依赖在现有独立环境中维护。

## 启动与运行

先启动 controller：

```bash
/opt/conda/envs/coreflow_mtp/bin/python -I -B scripts/system_start.py --port 18500
```

另一个终端按 allocation 启动实例；示例中的两张卡需预先空闲：

```bash
/opt/conda/envs/coreflow_mtp/bin/python -I -B scripts/start_instances_from_allocation.py start \
  --allocation examples/allocation.json --devices 0,2 \
  --controller-url http://127.0.0.1:18500 \
  --max-model-len 8192 --max-num-seqs 4 \
  --kv-cache-memory-bytes 1073741824 \
  --enable-prefix-caching --enable-cache-management --out-dir run/workflow_001
```

allocation 必须明确模型、agent、context 范围、实例数和 K。`--dry-run` 只查看命令。每批部署使用新目录，日志与进程创建身份记录在 `instances.json`；示例为手工配置，不表示优化器预测它最优。长任务在 tmux 中运行。

提交自己的完整 workflow JSON：

```bash
/opt/conda/envs/coreflow_mtp/bin/python -I -B scripts/run_workflow.py \
  --controller-url http://127.0.0.1:18500 \
  --payload /path/to/workflow.json --output run/workflow_001/workflow_result.json
```

节点可提供完整 `prompt_token_ids`，也可通过 `input_segments` 引用父节点本次实际输出。`parent_context` 可引用已完成祖先的完整上下文；不能引用无依赖关系的节点。不会用历史参考输出强制模型生成。纯长度 trace 不能用于 MTP 接受率；`replay_ar_trace.py` 的显式 synthetic 回放仅允许 K=0。

停止本批实例：

```bash
/opt/conda/envs/coreflow_mtp/bin/python -I -B scripts/start_instances_from_allocation.py stop \
  --state-file run/workflow_001/instances.json
```

停止器只处理本次记录且进程创建时间匹配的进程组，不应使用 `pkill python`。

## Profiling 与资源分配

| 入口 | 用途 |
| --- | --- |
| `model_profile.py` | 新环境 K=0 Llama 单层 attention、非 attention、干扰和拷贝 profiling |
| `verify_ar_profile.py` | CPU 侧组件拟合／留 context 验证 |
| `allocate_ar.py` | 经来源核验的 AR profile → 容量预测 → 分区／全局分配 |
| `allocate.py` | 读取匹配执行配置的容量表，按区间选择 K 后分区和分配 |
| `serve_instance.py` | 独立固定 K 实例 |
| `system_start.py` | controller |
| `start_instances_from_allocation.py` | allocation 到实例部署 |
| `run_workflow.py` | 提交完整 workflow |
| `replay_ar_trace.py` | 显式标记的旧长度 trace 回放，仅 K=0 |

各入口可用 `--help` 查看参数。K=0 自动组件模型目前限定 Llama、TP=1、FA2；Qwen3.5 hybrid/MTP 必须重新建模，不能套用旧 Llama profile、KV容量或独立 drafter 的系数。

容量表中的单位为按 workload 需求归一化的 agent invocation/s，不是 token/s。原 context DP／全局贪心结构继续使用，尚不表示 MTP 成本、产出、prefill、缓存与显存约束的联合容量预测已经完成。

## 缓存、路由与监控

- 显式 GPU 驻留：`keep_cache_in_gpu=true, free_cache=false`。
- 完成请求保存 CPU：`keep_cache_in_gpu=false, free_cache=false`。
- 最终释放：`free_cache=true`。需提供 query/invocation ID，并检查实际 `cache.stored`。
- 可选预测式策略：`--cache-policy-profile PATH`，请求 `cache_policy=predictive`；工具观测通过 `/tools/observe` 上报，或使用 `tools.observe_tool_call`。
- 默认严格 context 区间路由；原缓存优先兼容模式需 controller `--routing-policy coreflow` 与实例 `--cache-affinity-routing`，外区间复用必须有实际原生命中。
- `/metrics` 提供实际请求、token、缓存及调度 batch 指标；`--execution-trace-dir` 为可选同步诊断，会扰动计时，不作为未插桩性能。

缓存迁移仅限同机、相同模型／K／TP 的完成请求可复用前缀；不支持运行中请求、跨 K 或跨主机状态转换。细节见 [缓存协议](docs/缓存功能迁移.md) 与 [系统设计](docs/迁移设计.md)。服务默认 loopback、无鉴权，不应暴露公网。
