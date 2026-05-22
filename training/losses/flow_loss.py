import mlx.core as mx

def compute_flow_matching_loss(flow_matcher, z_target: mx.array, h_context: mx.array, mask: mx.array = None):
    """
    最优传输流匹配损失 (Rectified Flow Matching Loss)。
    这是整个系统唯一且终极的因果评判标准。
    
    Args:
        flow_matcher: 实例化的 FlowMatcher 喷嘴网络
        z_target: (B, L, d_model) 真实世界里下一句话的算子坐标 (来自 Phase 0.5)
        h_context: (B, L, d_model) Mamba 根据历史计算出的当前势能状态
        mask: (B, L) bool 类型的序列掩码，过滤掉 Padding
    """
    B, L, D = z_target.shape
    
    t = mx.random.uniform(shape=(B, L, 1))
    x_0 = mx.random.normal(shape=(B, L, D))
    x_t = (1.0 - t) * x_0 + t * z_target
    v_target = z_target - x_0
    
    v_pred = flow_matcher(x_t, t, h_context)
    
    # 计算逐点动能误差 (B, L, D) -> (B, L)
    # 核心改动：用 mean 取代 sum，让误差降到 ~1.2 左右，否则 1024 维的误差累加会产生破坏性的巨大梯度！
    squared_error = mx.mean(mx.square(v_pred - v_target), axis=-1)
    
    # 遮蔽 Padding 部分的误差 (极度安全写法，防止 NaN * 0 = NaN 导致全盘崩溃)
    if mask is not None:
        squared_error = mx.where(mask == 0, mx.zeros_like(squared_error), squared_error)
        loss = mx.sum(squared_error) / mx.maximum(mx.sum(mask), 1.0)
    else:
        loss = mx.mean(squared_error)
        
    return loss
