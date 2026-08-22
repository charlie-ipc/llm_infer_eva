# 640 卡 MoE Decode 系统性能估算说明

## 1. 文档目的

本文说明一个 640 卡、decode-only 的 MoE 推理部署估算，重点回答：

- batch=1024 如何分配到每个模型副本和每个通信 rank；
- TP reduce、EP dispatch、EP combine 分别传输多少数据；
- intra-node 带宽、链路延迟和 collective 启动延迟如何进入计算；
- 为什么 intra-node 带宽从 100 GB/s 增加到 800 GB/s 后，性能增幅有限；
- reduce 结果是否已分发到组内芯片，以及是否需要额外 broadcast；
- 当前结论依赖哪些尚需实机验证的假设。

本文是解析模型，不是实机 benchmark 或 P99 服务性能承诺。可执行复现入口是
[`examples/analysis/scan_640_card_intra_node.py`](../examples/analysis/scan_640_card_intra_node.py)。

## 2. 固定输入

### 2.1 模型和 workload

| 项目 | 数值 |
|---|---:|
| 模型类型 | synthetic MoE decoder-only |
| 总参数 | 约 10T |
| 每 token 激活参数 | 约 50B |
| 网络层数 | 80 |
| hidden size | 8192 |
| routed experts | 1024 |
| 每 token 选择专家数 top-k | 4 |
| shared experts | 0 |
| decode context | 4096 tokens，即 4095 历史 token + 1 个新 token |
| 系统 batch | 1024 |
| 目标单用户性能 | 30 token/s |
| 目标系统性能 | 1024 * 30 = 30,720 token/s |

`shared experts=0` 是本次估算的重要输入。模型只计算 attention TP 和 routed-expert
TP，不增加 shared-expert TP collective。

### 2.2 单卡加速器

| 项目 | 数值 |
|---|---:|
| GEMM 算力 | 1024 TOPS |
| VECTOR 算力 | 32 TOPS |
| DRAM 带宽 | 2000 GB/s，即 2 TB/s |
| DRAM 容量 | 200 GB |
| 每节点卡数 | 8 |
| intra-node latency | 1 us |
| inter-node bandwidth | 800 GB/s，扫描时固定 |
| inter-node latency | 5 us |
| collective launch latency | 8 us |

代码中的字段名是 `*_gbps`，但计算单位是十进制 GB/s，即带宽乘以 `1e9`
得到 bytes/s，不是 Gbit/s，也不是 GiB/s。

### 2.3 精度和通信格式

| 数据 | 位宽 | 用途 |
|---|---:|---|
| 权重 / activation / KV cache | FP4，4 bit | W4A4、KV4 和本地算子流量 |
| TP reduce payload | FP32，32 bit | attention 和 routed MoE 的 TP all-reduce |
| EP dispatch payload | FP4，4 bit | token activation 发送到专家 owner |
| EP combine payload | BF16，16 bit | top-k 专家输出返回并做本地加权合并 |

`tp_reduce_bits`、`ep_dispatch_bits`、`ep_combine_bits` 是独立参数，不能再由
activation 位宽隐式代替。

## 3. 640 卡并行拓扑

采用的固定方案为：

```text
total_cards = 640
cards_per_node = 8
nodes = 640 / 8 = 80

replicas = 16
cards_per_replica = 40
batch_per_replica = 64

attention_tp = 4
attention_dp = 10
moe_tp = 5
expert_parallel = 8
```

并行宽度满足：

```text
attention_tp * attention_dp = 4 * 10 = 40 cards/replica
moe_tp * expert_parallel = 5 * 8 = 40 cards/replica
replicas * cards_per_replica = 16 * 40 = 640 cards
replicas * batch_per_replica = 16 * 64 = 1024 requests
```

这里的 `attention_dp` 是一个模型副本内部对请求的切分，不是 16 个完整模型副本
之间的数据并行。16 个 replica 独立执行，不建模跨 replica 的同步 collective。

## 4. Batch 如何进入通信量

### 4.1 Attention rank 的最忙 batch

每个副本有 64 个请求，由 10 个 attention-DP rank 分担。64 不能被 10 整除，
容量和时延按最忙 rank 保守估算：

```text
local_attention_requests = ceil(64 / 10) = 7
```

因此 attention TP 和 EP payload 使用 7 个本地请求，而不是忽略 batch，也不是直接
使用系统总 batch=1024。

### 4.2 Routed expert rank 的 token assignment

每个副本的一次 decode step 产生 64 个 token；每个 token 路由到 4 个专家，共有：

```text
routed assignments per replica = 64 * 4 = 256
local routed assignments = ceil(256 / 8 EP ranks) = 32
```

这 32 个 assignment 是 routed MoE TP all-reduce 的最忙 rank 输入。

## 5. 每次 Collective 的 Payload

统一公式为：

```text
payload_bytes = elements * bits / 8
```

### 5.1 Attention TP reduce

```text
elements = 7 requests * 8192 hidden = 57,344
payload = 57,344 * 32 / 8 = 229,376 bytes
group size = attention_tp = 4
```

### 5.2 Routed MoE TP reduce

```text
elements = 32 assignments * 8192 hidden = 262,144
payload = 262,144 * 32 / 8 = 1,048,576 bytes
group size = moe_tp = 5
```

### 5.3 EP dispatch

```text
elements = 7 requests * 4 experts/token * 8192 hidden = 229,376
payload = 229,376 * 4 / 8 = 114,688 bytes
group size = expert_parallel = 8
```

### 5.4 EP combine

```text
elements = 7 requests * 4 experts/token * 8192 hidden = 229,376
payload = 229,376 * 16 / 8 = 458,752 bytes
group size = expert_parallel = 8
```

EP combine 按 BF16 统计所有 top-k 专家输出的返回流量。专家权重系数和本地加权求和
属于 VECTOR 计算；模型不为本地求和额外增加网络流量。之后的 routed MoE TP
all-reduce 使用 FP32 payload，使 TP 组内各 rank 获得一致输出。

## 6. Collective 原理和公式

### 6.1 链路选择

当前实现假设 rank 紧凑放置：

```text
group_size <= cards_per_node  -> intra_node
group_size >  cards_per_node  -> inter_node
```

本方案的 group size 分别是 attention TP=4、MoE TP=5、EP=8，均不超过每节点
8 卡，因此四个 collective 全部选择 intra-node。扫描 inter-node 带宽不会改变该
方案的结果；需要扫描的是 `intra_node_gbps`。

这个规则只按 group size 选路，不验证 40 卡副本中每个逻辑组的实际物理 rank
映射，也不建模交换网络争用。专家需要确认四类组能否同时按该假设放置。

### 6.2 Ring all-reduce

实现使用 ring reduce-scatter + all-gather 的解析近似：

```text
transfer_bytes = payload * 2 * (N - 1) / N
latency_steps = 2 * (N - 1)
time = transfer_bytes / (bandwidth_GBps * 1e9)
     + latency_steps * link_latency_us / 1e6
     + collective_launch_latency_us / 1e6
```

all-reduce 完成后，reduced activation 已通过 all-gather 分发到组内所有 rank。
因此不再单独增加 broadcast；若再加一次 broadcast 会重复统计。仓库仍提供独立
`broadcast_cost`，用于真正只有单源广播语义的其他操作。

### 6.3 Ring all-to-all

EP dispatch 和 combine 使用 ring all-to-all 近似：

```text
transfer_bytes = payload * (N - 1) / N
latency_steps = N - 1
time = transfer_bytes / (bandwidth_GBps * 1e9)
     + latency_steps * link_latency_us / 1e6
     + collective_launch_latency_us / 1e6
```

### 6.4 逐层聚合

80 层中，每层计算：

```text
1 * attention TP all-reduce
1 * routed MoE TP all-reduce
1 * EP dispatch all-to-all
1 * EP combine all-to-all
```

因此一次 decode step 有 320 次 modeled collective launch。本模型把 GEMM、VECTOR、
TP 和 EP 时间串行相加，不假设通信与计算重叠。

## 7. 800 GB/s 的逐项算例

### 7.1 每层和 80 层通信时间

| Collective | Transfer bytes | 固定 latency + launch / 层 | 带宽时间 / 层 | 80 层时间 |
|---|---:|---:|---:|---:|
| Attention TP AR, N=4 | 344,064 | 14 us | 0.430080 us | 1.154406 ms |
| Routed MoE TP AR, N=5 | 1,677,721.6 | 16 us | 2.097152 us | 1.447772 ms |
| EP dispatch A2A, N=8 | 100,352 | 15 us | 0.125440 us | 1.210035 ms |
| EP combine A2A, N=8 | 401,408 | 15 us | 0.501760 us | 1.240141 ms |

所以：

```text
TP = 1.1544064 + 1.44777216 = 2.60217856 ms
EP = 1.2100352 + 1.2401408 = 2.45017600 ms
communication total = 5.05235456 ms
```

固定计算项来自已接受的 640 卡结果：

```text
GEMM = 20.801879 ms
VECTOR = 1.949727 ms
compute total = 22.751606 ms
```

最终：

```text
TPOT = 22.751606 + 5.05235456 = 27.80396056 ms
single-user rate = 1000 / 27.80396056 = 35.966099 token/s
system rate = 1024 * 35.966099 = 36,829.285 token/s
```

30 token/s 对应最大 TPOT 为 `1000/30 = 33.333333 ms`。800 GB/s 时的 TPOT
余量约 5.529 ms，吞吐余量约 19.89%。

## 8. Intra-node 带宽扫描

下面的结果由复现脚本直接生成，inter-node 固定为 800 GB/s：

| Intra-node GB/s | TP ms | EP ms | Total ms | User token/s | System token/s |
|---:|---:|---:|---:|---:|---:|
| 100 | 4.017428 | 2.801408 | 29.570442 | 33.817553 | 34,629.174 |
| 200 | 3.208714 | 2.600704 | 28.561024 | 35.012750 | 35,853.056 |
| 300 | 2.939143 | 2.533803 | 28.224551 | 35.430147 | 36,280.470 |
| 400 | 2.804357 | 2.500352 | 28.056315 | 35.642599 | 36,498.022 |
| 500 | 2.723486 | 2.480282 | 27.955373 | 35.771298 | 36,629.810 |
| 600 | 2.669571 | 2.466901 | 27.888079 | 35.857615 | 36,718.198 |
| 700 | 2.631061 | 2.457344 | 27.840011 | 35.919526 | 36,781.594 |
| 800 | 2.602179 | 2.450176 | 27.803961 | 35.966099 | 36,829.285 |

即使 intra-node 只有 100 GB/s，结果仍为 33.817553 token/s，比 30 token/s
目标高约 12.73%。从 100 提升到 800 GB/s，单用户性能增加约 6.35%。

## 9. 为什么带宽收益有限

在 800 GB/s 下：

```text
collective 固定 latency + launch = 4.800000 ms
collective 带宽传输时间 = 0.25235456 ms
communication total = 5.05235456 ms
```

即固定项占通信时间约 95%。就算带宽无限大，当前串行模型的理论下限仍为：

```text
minimum TPOT = GEMM 20.801879
             + VECTOR 1.949727
             + fixed communication 4.800000
             = 27.551606 ms
maximum user rate = 36.295525 token/s
```

所以 800 GB/s 已接近本解析模型的带宽上限。若希望继续提升，应优先审查 collective
合并、启动开销、层间 pipeline、通信计算重叠，以及 GEMM/VECTOR 实际效率，而不仅是
继续增加 nominal bandwidth。

## 10. 与历史 CSV 的关系

[`results/decode_640_cards.csv`](../results/decode_640_cards.csv) 保留初始搜索结果：

```text
TP = 2.425272 ms
EP = 2.420070 ms
TPOT = 27.596948 ms
user rate = 36.235890 token/s
```

该结果生成时通信尚未拆分独立位宽，TP、EP dispatch 和 EP combine 都沿用 4-bit
activation。用当前 collective 公式和旧位宽可以逐项复现这些历史值。

当前口径将 TP reduce 改为 FP32、EP combine 改为 BF16，因此 800 GB/s 结果更新为
35.966099 token/s。历史 CSV 不被覆盖，以便审查模型演进。

在准备本文时还纠正了一次临时扫描错误：此前临时 Python 片段误加了一个
shared-expert TP all-reduce，得到约 34.339880 token/s；目标模型没有 shared expert，
而且该额外 collective 无法复现历史 CSV 的 `tp_ms`，所以不属于本方案。

## 11. 当前未建模项

- 没有按实际机箱、交换芯片和 rank mapping 建立拓扑图。
- 没有网络效率系数、协议头、重传、拥塞、双工方向或交换机超卖。
- 多个 TP/EP group 同时通信，默认各自获得完整 nominal bandwidth。
- GEMM、VECTOR、TP、EP 串行相加，不考虑 overlap。
- collective 使用 ring 解析式，不区分 tree、hierarchical 或硬件 multicast。
- 每层固定一次每类 collective，不做 kernel fusion 或跨层合并。
- 使用平均解析时延，不包含调度、排队、抖动和 P99/P999。
- GEMM/VECTOR 项固定自历史方案，本次扫描只隔离通信带宽，不重新搜索并行计划。
- 参数/KV 容量沿用历史方案的 149.445 GB/卡估算，本脚本不重新计算内存。

## 12. 专家建议重点

1. 40 卡副本中，TP=4、MoE TP=5、EP=8 是否能同时紧凑映射在 8 卡节点内。
2. 目标 collective 库实际采用 ring、tree 还是 hierarchical 算法。
3. TP reduce 是否全程 FP32，还是 BF16/FP16 传输、FP32 局部累加。
4. EP dispatch FP4 和 EP combine BF16 是否与目标 kernel/网络协议一致。
5. EP combine 是否传回全部 top-k hidden，还是在专家侧/网络侧提前归并。
6. 8 us launch 和 1 us intra-node latency 是否来自目标平台实测。
7. 同时运行的 collective 是否共享链路，实际有效带宽应乘多少效率系数。
8. 通信与 GEMM/VECTOR 能否重叠，重叠比例如何标定。
9. 30 token/s 是否要求平均值、P50、P99，是否需加入 scheduler 和服务抖动。

专家确认这些输入后，才应把 640 卡结论作为工程容量规划依据。
