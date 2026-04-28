# HollyShit: 内生自举的三段式流体力学模型架构

将语言表达视为语义空间中的连续流体力学传播，彻底解耦“静态表征”、“上下文状态”与“动态轨迹预测”。本系统致力于从对外部预训练巨头模型的“寄生”，走向完全的“内生自举（Endogenous Bootstrapping）”。

## 阶段演进：从测绘到内生

### Phase 0: 神之测绘 (GodEncoder 训练)
* **核心目标**：建立系统内部绝对的“物理坐标系”（真理语义空间）。
* **机制与逻辑**：
  * 同时吃进 5 个外部顶尖 Embedding 模型（BGE, Qwen, Xiaobu, Youtu, Conan）的特征，通过 `SensoryFuser` 进行 LayerNorm 降维与质心融合。
  * 引入一个无记忆的极简重构器（`WeakDecoder`），通过交叉熵和 BoW 损失，倒逼 `GodEncoder` 将这 5 个模型的高维共识坍缩到一个绝对的物理坐标 $z_{target}$ 上。
* **物理意义**：这一步在给宇宙打钢印，为后续所有的动态预测和上下文编码提供绝对的真理参考体系。

### Phase 0.5: 正向蒸馏 (Inward Latent Distillation)
* **核心目标**：彻底踢开沉重的外部大模型，打造完全独立的**“静态视网膜” (TinyCharEncoder)**。
* **机制与逻辑**：
  * 使用海量的纯文本作为输入。
  * 训练一个原生的、基于单字（Char）的轻量级小编码器，用最简单暴力的 **MSE Loss** 强行拟合 Phase 0 生成的数百万个 $z_{target}$ 绝对坐标。
* **物理意义**：一旦小模型练成，它就彻底“白嫖”并锁死了 5 大巨头的联合空间拓扑法则。部署时输入生肉字符，它就能瞬间指明该概念的空间坐标，系统就此实现 100% 物理脱钩与自举。

*(后续架构演进包括 Phase 1: Mamba 纯上下文大脑训练，以及 Phase 2: Flow Matching 微观动力学推演。详见 `designs/mindstorm.md`)*

---

## 运行步骤指南

### Step 1: Phase 0 绝对坐标系确立 (Mac/MLX 本地)
在本地 Apple Silicon 环境下，执行多模态特征融合，训练 GodEncoder 生成 $z_{target}$ 坐标点阵。
```bash
# 执行多模型联合训练
python training/train_phase0_v2.py \
    --config_file dataset_config.json \
    --batch_size 128 \
    --epochs 3
```

### Step 2: Phase 0.5 静态视网膜蒸馏 (云端/CUDA)
由于需要极大规模的单字吞吐以对齐拓扑空间，该阶段在云端（PyTorch/CUDA）作为子任务独立执行。
```bash
# 进入蒸馏子仓库
cd distilled_emb

# 启动 PyTorch 蒸馏，将生肉文本强行拟合到 Phase 0 生成的绝对物理坐标上
python train_cuda.py \
    --p0_ckpt ../checkpoints/run/[你的Phase0权重目录] \
    --batch_size 256 \
    --epochs 5 \
    --out_dir checkpoints/distilled
```

### Step 3: 下载回流与 Phase 1 准备
蒸馏完成后，将云端生成的 `TinyCharEncoder` (视网膜) 权重下载回本地 Mac。

接下来，我们将编写转换脚本将 PyTorch 权重无损转为 MLX 格式。在未来的 Phase 1 中，这双纯内生的视网膜将全面替换掉沉重缓慢的 5 模型外部特征流，正式喂给 Mamba 启动上下文大脑的训练。
