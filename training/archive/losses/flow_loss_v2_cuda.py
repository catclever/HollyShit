# =========================================================================
# [ARCHIVED] LEGACY SCRIPT: Anisotropic Flow Loss (V2)
#
# [Reason for Archival]:
# This loss function introduced an Anisotropic Prior (using empirical mean and std
# from the Z manifold) to optimize the ODE trajectory. While mathematically sound 
# for a single point, it was designed for the "Point-based Flow Matching" architecture.
# Because predicting a single pooled Z vector loses exact sequence token ordering, 
# this loss is being retired in favor of a Sequence-based Flow Matching loss.
# =========================================================================

import torch

def compute_flow_matching_loss_v2(flow_matcher, z_target: torch.Tensor, h_context: torch.Tensor, 
                                 prior_mean: torch.Tensor, prior_std: torch.Tensor,
                                 mask: torch.Tensor = None, cfg_drop_prob: float = 0.1):
    """
    最优传输流匹配损失 (Rectified Flow Matching Loss) 的 V2 进化版本。
    【新增能力】：支持各向异性高斯先验 (Anisotropic Gaussian Prior)
    
    Args:
        flow_matcher: 实例化的 FlowMatcher 喷嘴网络
        z_target: (B, L, d_model) 真实世界里下一句话的算子坐标 (来自 Phase 0.5)
        h_context: (B, L, d_model) Mamba 根据历史计算出的当前势能状态
        prior_mean: (d_model,) 预计算的全局特征均值
        prior_std: (d_model,) 预计算的全局特征标准差
        mask: (B, L) bool 类型的序列掩码，过滤掉 Padding
        cfg_drop_prob: 训练时以一定概率将 h_context 置零，为无分类器引导(CFG)做准备
    """
    B, L, D = z_target.shape
    device = z_target.device
    dtype = z_target.dtype
    
    # 模拟 CFG Context Dropout
    if cfg_drop_prob > 0.0:
        drop_mask = torch.rand(size=(B, 1, 1), device=device, dtype=dtype) > cfg_drop_prob
        h_context = torch.where(drop_mask, h_context, torch.zeros_like(h_context))
        
    t = torch.rand(size=(B, L, 1), device=device, dtype=dtype)
    
    # 核心修改点：引入空间规范化
    # 原始的各向同性高斯，强行拉伸并平移至目标流形的宏观外壳大小
    x_0_base = torch.randn(size=(B, L, D), device=device, dtype=dtype)
    x_0 = x_0_base * prior_std.to(dtype) + prior_mean.to(dtype)
    
    # 物理积分公式 (直线输运)
    x_t = (1.0 - t) * x_0 + t * z_target
    v_target = z_target - x_0
    
    # 网络预测
    v_pred = flow_matcher(x_t, t, h_context)
    
    # 计算逐点动能误差 (B, L, D) -> (B, L)
    # 注意: 这里用的是 mean(dim=-1)，这是保证 Loss 不受 D 维度暴涨影响的关键
    squared_error = torch.mean(torch.square(v_pred - v_target), dim=-1)
    
    # 遮蔽 Padding 部分的误差 (极度安全写法，防止 NaN * 0 = NaN 导致全盘崩溃)
    if mask is not None:
        squared_error = torch.where(mask == 0, torch.zeros_like(squared_error), squared_error)
        loss = torch.sum(squared_error) / torch.clamp(torch.sum(mask), min=1.0)
    else:
        loss = torch.mean(squared_error)
        
    return loss
