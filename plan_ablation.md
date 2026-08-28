# Phase 1.5 Ablation & Super-Batch Plan

## 1. 架构做减法 (模型层面)
**目标文件**: `model/phase1_5_conceptdecoder.py`
- **移除记忆锁链**: 删除 `decoder_window=32` 和滑窗掩码逻辑，恢复标准的、全量历史的因果注意力掩码 (`Causal Mask`)，让模型重新拥有理解长句子的能力。
- **拆除交叉注意力**: 删除 `zoh_cross_attn` 层。由于 ZOH 的序列长度与文本序列长度完美对齐，我们将采用**直接残差相加** (`输入 = 字符Embed + ZOH概念特征`)，极其快速且暴力地将概念注入底层。
- **保留双头门控 (Option B)**: 保留指针网络的逻辑。模型将输出两个概率：一个是标准的生成概率，一个是 $p_{gen}$ 门控值，用于计算拷贝 `anchors` 的权重。

## 2. 训练流改造 (超级批次机制 Super-Batch)
**目标文件**: `training/train_phase1_5_conceptdecoder.py`
- **问题**: 目前是在每个 Mini-Batch (16条数据) 内部，频繁进行 GPU(编码) -> CPU(算锚点) -> GPU(解码) 的切换，严重阻断了流水线。
- **改造方案 (超级批次缓冲)**:
  1. 每次从数据集中读取一个**超级批次 (Super-Batch)**，例如 256 条文档。
  2. 集中让 GPU 跑完这 256 条文档的 `BaseEncoder` 和 `Chunker`，得到 `mx_c`。
  3. 将 `mx_c` 统一拉回 CPU，一次性并行/集中算好这 256 条文档的锚点 (`anchors_h`, `anchors_t`)。
  4. 将这 256 条准备好的数据，切分成 16 个 Mini-Batch (每个 16 条)，**连续、毫无停顿地喂给 Decoder 所在的 GPU 进行训练**。
- **收益**: 极大地平摊了 CPU 计算锚点的等待时间，GPU 在执行 16 次解码器反向传播时将处于纯粹的“满载饥渴”状态，效率将获得质的飞跃。

## 3. 执行步骤
1. 修改 `model/phase1_5_conceptdecoder.py`。
2. 重写 `train_phase1_5_conceptdecoder.py` 的数据迭代与训练循环逻辑。
3. 冒烟测试并重新开始训练。
