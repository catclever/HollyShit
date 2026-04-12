import mlx.core as mx
import math

def ot_cfm_loss(model, token_ids: mx.array, z_target: mx.array, mask: mx.array = None,
                x1_weight: float = 0.0, snap_ce_weight: float = 0.0,
                t_power: float = 1.0, align_threshold: float = 0.6, align_max_n: int = 12):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) Loss with auxiliary objectives.
    
    Args:
        model: FlowDecoder instance
        token_ids: (B, L) ground truth token IDs
        z_target: (B, z_dim) semantic anchor from GodEncoder
        mask: (B, L) padding mask (1=valid, 0=pad)
        x1_weight: Weight for x₁-consistency loss (recommended 0.5~1.0)
        snap_ce_weight: Weight for snap cross-entropy loss (recommended 0.1~0.5)
        t_power: Power for high-t sampling bias. 1.0 = uniform, <1.0 = bias towards t≈1
                 (recommended 0.5 for sqrt bias)
        align_threshold: Cosine similarity threshold to start rewarding continuous N-Grams (e.g. 0.6 for 53°, 0.7 for 45°).
        align_max_n: Maximum length to search for continuous sequence alignments.
    
    Returns:
        total_loss, loss_dict  (loss_dict for logging)
    """
    B, L = token_ids.shape
    
    # 1. Ground Truth Destination (The physical coordinates of the sentence)
    x_1 = model.embed_text(token_ids)  # (B, L, d_model)
    
    # 2. Primordial Chaos (Standard Normal Distribution)
    x_0 = mx.random.normal(shape=x_1.shape)  # (B, L, d_model)
    
    # 3. Time Sampling with optional high-t bias
    # t_power < 1.0 → t = u^power, biases towards t≈1 (last-mile precision)
    # t_power = 1.0 → uniform (original behavior)
    # t_power = 0.5 → sqrt bias, ~70% of samples have t > 0.5
    u = mx.random.uniform(shape=(B, 1))
    t = mx.power(u, t_power)
    
    # 4. Interpolate the exact intermediate state
    t_expand = mx.expand_dims(t, 2)  # (B, 1, 1)
    x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
    
    # 5. Base Model Prediction (先求出无导向的假想速度)
    v_pred = model(x_t, t, z_target, mask=mask)
    
    # ====================================================================
    # ====== 核心升级：Dynamic N-Gram Flow Alignment (动态多路软流向) ======
    # ====================================================================
    # 预测模型当前假想的降落坐标
    x1_pred = x_t + (1.0 - t_expand) * v_pred  # (B, L, d_model)
    
    # 算当前预测位置与真实序列所有字的"绝对余弦相似度" (取代绝对 0/1 正确与否)
    norm_pred = mx.sqrt(mx.sum(mx.square(x1_pred), axis=-1, keepdims=True) + 1e-8)
    norm_true = mx.sqrt(mx.sum(mx.square(x_1), axis=-1, keepdims=True) + 1e-8)
    cos_sim = mx.matmul(x1_pred / norm_pred, mx.transpose(x_1 / norm_true, (0, 2, 1)))  # [-1, 1]
    
    # 定义相似度阈值并生成浮点掩码
    float_mask = (cos_sim > align_threshold).astype(mx.float32)
    
    # 基础点积势能场 (保留原始规模，不限制在[-1,1]以用于真实的流场梯度传递)
    D_val = x_1.shape[-1]
    sim = mx.matmul(x1_pred, mx.transpose(x_1, (0, 2, 1))) / math.sqrt(D_val)  # (B, L, L)
    
    ngram_sim = sim
    current_shift = sim
    chain_valid = None
    multiplier = 1.2
    
    # 动态长序列连续暴击叠加 (断开即停止叠加)
    L_target = x_1.shape[1]
    # 为保障 MLX 计算图极速编译不发生阻塞，我们设置合理的向前探测极限上限 (利用 align_max_n)
    # 因为由 float_mask 控制了连贯断流，不足最大长度的链会在半途直接自动触发乘以 0.0 而被切断
    max_k = min(L_target, align_max_n) 
    for k in range(1, max_k):
        # 矩阵物理左上推移
        current_shift = mx.pad(current_shift[:, :-1, :-1], [(0,0), (1,0), (1,0)])
        float_mask_shifted = mx.pad(float_mask[:, :-1, :-1], [(0,0), (1,0), (1,0)])
        
        # 链条连贯性检验：只有当前位 > threshold，且之前的所有位也 > threshold，链条才算存活
        if k == 1:
            chain_valid = float_mask_shifted
        else:
            chain_valid = chain_valid * float_mask_shifted
            
        # 存活链条的倍数奖励累加 (全为 0 的链条会自动失效不影响结果)
        ngram_sim = ngram_sim + (multiplier ** k) * current_shift * chain_valid
        
        # 更新掩码用于下一步的漂移相乘
        float_mask = float_mask_shifted
    
    # 多序列软坍缩 (Soft-Max Assignment)
    # 不强制 1V1 锁死，允许当前点受多个长序列引力的叠加拉扯
    align_weights = mx.softmax(ngram_sim, axis=-1)  # (B, L, L)
    
    # 动态重构物理靶点
    raw_dynamic_x1 = mx.matmul(align_weights, x_1)  # (B, L, d_model)
    
    # 🚨 物理奇点修复机制 (Singularity Fade-out) 🚨
    # 在最后降落阶段 (t -> 1)，任何非真实的 raw_dynamic_x1 都会导致物理上需要“瞬间光速折返”的奇异点
    # 这里通过极其平滑的倒数引力，让动态重组逐渐向真实的单体 x_1 妥协。
    # 这样，在 t=0.99 时，靶中心必然无限折回 x_1，分子分母同阶消失，彻底避免 Loss 在终点处除零爆炸并达到完美吻合！
    dynamic_x1 = raw_dynamic_x1 * (1.0 - t_expand) + x_1 * t_expand
    
    # 6. Optimal Transport Velocity (由静态的终点，变为被动态锚点吸过去的引力线)
    # 计算当前从 x_t 抵达妥协后的 dynamic_x1 所需要的瞬时速度：
    v_true_dynamic = (dynamic_x1 - x_t) / mx.maximum(1.0 - t_expand, 1e-5)
    
    # ====== Loss 1: Dynamic Velocity MSE ======
    mse = mx.square(v_pred - v_true_dynamic).mean(axis=-1)  # (B, L)
    
    if mask is not None:
        num_valid = mx.maximum(mask.sum(), 1.0)
        L_velocity = (mse * mask).sum() / num_valid
    else:
        num_valid = B * L
        L_velocity = mse.mean()
    
    total_loss = L_velocity
    
    # ====== Loss 2: x₁-Consistency (does the predicted velocity point to the right destination?) ======
    L_x1 = mx.array(0.0)
    if x1_weight > 0:
        # From v_pred, reverse-engineer where the model thinks x₁ should be
        # x₁_pred = x_t + (1 - t) * v_pred
        x1_pred = x_t + (1.0 - t_expand) * v_pred
        x1_mse = mx.square(x1_pred - x_1).mean(axis=-1)  # (B, L)
        
        if mask is not None:
            L_x1 = (x1_mse * mask).sum() / num_valid
        else:
            L_x1 = x1_mse.mean()
        
        total_loss = total_loss + x1_weight * L_x1
    
    # ====== Loss 3: Snap Cross-Entropy (Lazy & Smooth CE Mechanism) ======
    L_snap = mx.array(0.0)
    if snap_ce_weight > 0:
        # 懒癌平滑过渡阀门 (Smooth Envelope)
        # 用意: 让模型只有在流的后期 (t>0.7) 时才逐渐感知到 CE Loss。
        # 避免如果在某一步突然随机引入 CE 而导致原本依靠距离优化的梯度方向“瞬间骨折突变”。
        # 当 t < 0.7 时，t_weight 完全为 0，不耗费任何梯度扰动；
        # 当 t 处于 0.7(起飞) 到 1.0(终点快照) 期间时，权重像斜坡一样从 0.0 渐平滑爬升到 1.0。
        t_weight = mx.clip((t_expand - 0.7) / 0.3, 0.0, 1.0)
        
        # Get the embedding bank
        # 对应的解码阶段也必须保证备用候选库（emb_bank）全部在同一个 L2 球面上进行点积比较
        raw_bank = model.char_embedding.weight  # (V, d_model)
        emb_bank = (raw_bank / mx.maximum(mx.linalg.norm(raw_bank, axis=-1, keepdims=True), 1e-6)) * math.sqrt(model.d_model)
        
        # Soft Temperature Logits
        logits = mx.matmul(x1_pred, mx.transpose(emb_bank, (1, 0))) / math.sqrt(model.d_model)
        
        # CE Computation
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = token_ids.reshape(-1)
        
        log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
        ce = -log_probs[mx.arange(targets_flat.shape[0]), targets_flat]  # (B*L,)
        
        # 将缓启动的斜坡权重加在每一个时间帧切片的 CE 惩罚上
        ce = ce * mx.broadcast_to(t_weight, (B, L, 1)).reshape(-1)
        
        if mask is not None:
            mask_flat = mask.reshape(-1)
            L_snap = (ce * mask_flat).sum() / num_valid
        else:
            L_snap = ce.mean()
        
        total_loss = total_loss + snap_ce_weight * L_snap
    
    return total_loss
